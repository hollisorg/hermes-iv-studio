"""IV Studio for Hermes: free, local video and picture generation on your own GPU.

Registers IV Studio (https://github.com/hollisorg/iv-studio) as a Hermes video-generation and image-generation
backend, so Hermes's own ``video_generate`` and ``image_generate`` tools render on the person's PC through ComfyUI
and MiniMax H3 instead of a paid cloud API, plus an ``iv_studio_status`` tool. See README.md for what it reads,
runs and stores.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ivclient import IVStudio, IVStudioError, find_command

logger = logging.getLogger(__name__)

NAME = "iv-studio"
DISPLAY = "IV Studio (your GPU, free)"
INSTALL_URL = "https://github.com/hollisorg/iv-studio/releases/latest"
H3_ASPECTS = ("16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3", "21:9")
POLL_S = 4.0
RENDER_DEADLINE_S = 40 * 60

# Hermes's resolution names -> how IV Studio renders and finishes one take.
#   480p: the preview take itself (about 864x480), the quickest.
#   720p: the take finished at 1 MP (about 1376x768) with DLSS 5.
#   1080p: the take finished at 1 MP, DLSS 5 and RTX 2x (about 2752x1536).
RESOLUTIONS = {
    "480p": {"finish": None},
    "540p": {"finish": None},
    "720p": {"finish": {"megapixels": 1, "upscale2x": False}},
    "768p": {"finish": {"megapixels": 1, "upscale2x": False}},
    "1080p": {"finish": {"megapixels": 1, "upscale2x": True}},
}

# Signs that a prompt is already written for H3 (IV Studio's checker reads it as it is); anything else is a plain
# idea and goes through IV Studio's prompt writer first.
_H3_MARKERS = re.compile(r"\[Shot \d+\]|<d>|<Picture \d+>|\{char\}|Subject \d+:", re.I)


# ---------------------------------------------------------------------------------------------------------------
# small helpers

_CTX: Any = None  # the PluginContext, kept for this plugin's own settings


def _exe_override() -> str:
    """The IV Studio executable from this plugin's settings (plugins.entries.iv-studio.settings.ivstudio_path), for
    an install somewhere unusual such as a portable copy. Empty: look in the usual places."""
    try:
        return str((_CTX.get_config("ivstudio_path", "") if _CTX else "") or "").strip()
    except Exception:
        return ""


def _data_dir() -> Path:
    try:
        from plugins.plugin_storage import plugin_data_dir
        d = Path(plugin_data_dir(NAME))
    except Exception:
        d = Path(tempfile.gettempdir()) / "hermes-iv-studio"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _local_file(ref: str, folder: Path) -> str:
    """A picture the person gave (a path, a file:// URL or an http(s) URL) as a local file IV Studio can read."""
    ref = (ref or "").strip()
    if ref.startswith("file://"):
        ref = urllib.parse.unquote(urllib.parse.urlparse(ref).path)
        if re.match(r"^/[A-Za-z]:/", ref):
            ref = ref[1:]
    if ref.startswith(("http://", "https://")):
        name = Path(urllib.parse.urlparse(ref).path).name or "picture.png"
        dest = folder / f"{int(time.time() * 1000)}-{re.sub(r'[^A-Za-z0-9._-]', '_', name)[-80:]}"
        req = urllib.request.Request(ref, headers={"User-Agent": "hermes-iv-studio/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:  # noqa: S310 (person's own URL)
            shutil.copyfileobj(r, f, length=1 << 20)
        return str(dest)
    p = Path(ref).expanduser()
    if not p.is_file():
        raise IVStudioError(f"Can't find the picture {ref}")
    return str(p.resolve())


def _fetch_output(out: dict[str, Any], folder: Path, prefix: str) -> str:
    """A finished IV Studio file as a local path: its own path when it has one, else a copy from ComfyUI."""
    if out.get("path") and Path(out["path"]).is_file():
        return str(out["path"])
    url = out.get("url")
    if not url:
        raise IVStudioError("IV Studio finished but gave no file.")
    if url.startswith(("http://", "https://")):
        ext = Path(out.get("filename") or "out.mp4").suffix or ".mp4"
        dest = folder / f"{prefix}-{int(time.time())}{ext}"
        with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:  # noqa: S310 (local ComfyUI)
            shutil.copyfileobj(r, f, length=1 << 20)
        return str(dest)
    return url


def _wait_job(iv: IVStudio, job_id: str, label: str) -> dict[str, Any]:
    deadline = time.monotonic() + RENDER_DEADLINE_S
    while True:
        job = iv.call("get_job", {"jobId": job_id}, timeout=90)
        status = job.get("status")
        if status == "success":
            return job
        if status in ("error", "cancelled", "canceled", "failed"):
            raise IVStudioError(f"{label} failed: {job.get('error') or status}")
        if time.monotonic() > deadline:
            raise IVStudioError(f"{label} is still {status} after {RENDER_DEADLINE_S // 60} minutes; it keeps going in IV Studio.")
        time.sleep(POLL_S)


def _pick_character(prompt: str, characters: List[dict[str, Any]], asked: Optional[str]) -> Optional[dict[str, Any]]:
    """The IV Studio character the prompt is about: one named in ``character``, else one whose name, first name or
    trigger word appears in the prompt (each character is a trained LoRA, so the likeness holds)."""
    if asked:
        a = asked.strip().lower()
        for c in characters:
            if a in (str(c.get("id", "")).lower(), str(c.get("name", "")).lower(), str(c.get("trigger", "")).lower()):
                return c
    low = prompt.lower()
    for c in characters:
        name = str(c.get("name") or "")
        words = [w for w in {name, name.split(" ")[0], str(c.get("trigger") or "")} if len(w) >= 3]
        if any(re.search(rf"\b{re.escape(w.lower())}\b", low) for w in words):
            return c
    return None


def _with_char(prompt: str, c: Optional[dict[str, Any]]) -> str:
    """Write ``{char}`` where the character is named, which IV Studio swaps for the LoRA's trigger word."""
    if not c or "{char}" in prompt:
        return prompt
    name = str(c.get("name") or "")
    for w in sorted({name, name.split(" ")[0], str(c.get("trigger") or "")}, key=len, reverse=True):
        if len(w) >= 3:
            prompt, n = re.subn(rf"\b{re.escape(w)}\b", "{char}", prompt, count=1, flags=re.I)
            if n:
                return prompt
    return prompt


def _status(exe_override: str = "") -> dict[str, Any]:
    if not find_command(exe_override):
        return {"installed": False, "ready": False, "say": f"IV Studio isn't installed. Get it free: {INSTALL_URL}"}
    with IVStudio(exe_override) as iv:
        setup = iv.call("check_setup", timeout=90)
        items = setup.get("items", []) if isinstance(setup, dict) else []
        problems = [f"{i.get('label')}: {i.get('detail')}" for i in items if i.get("status") not in ("ok", "optional") and i.get("kind") != "optional"]
        comfy = next((i.get("detail") for i in items if i.get("id") == "comfyui"), None)
        gpu = next((i.get("detail") for i in items if i.get("id") == "gpu"), None)
        chars = iv.call("list_characters")
        speeds = iv.call("get_speeds")
    return {
        "installed": True,
        "ready": bool(setup.get("ready")) if isinstance(setup, dict) else False,
        "comfyui": comfy,
        "gpu": gpu,
        "speed": speeds.get("saved") if isinstance(speeds, dict) else None,
        "characters": [{"id": c.get("id"), "name": c.get("name")} for c in (chars or [])],
        "needs": problems[:8],
    }


# ---------------------------------------------------------------------------------------------------------------
# video

def _video_provider_class():
    from agent.video_gen_provider import VideoGenProvider, error_response, success_response

    class IVStudioVideo(VideoGenProvider):
        """MiniMax H3 on the person's own GPU, through IV Studio and ComfyUI. Free; one take per call."""

        @property
        def name(self) -> str:
            return NAME

        @property
        def display_name(self) -> str:
            return DISPLAY

        def is_available(self) -> bool:
            return find_command(_exe_override()) is not None

        def list_models(self) -> List[Dict[str, Any]]:
            return [{
                "id": "minimax-h3",
                "display": "MiniMax H3 (local)",
                "speed": "about 2-6 min on an RTX 4090",
                "strengths": "Sound and speech in the same render, IV Studio characters (trained LoRAs) by name, first-frame and reference pictures.",
                "price": "free (your own GPU)",
                "modalities": ["text", "image"],
            }]

        def default_model(self) -> Optional[str]:
            return "minimax-h3"

        def capabilities(self) -> Dict[str, Any]:
            return {
                "modalities": ["text", "image"],
                "aspect_ratios": list(H3_ASPECTS),
                "resolutions": ["480p", "720p", "1080p"],
                "min_duration": 1,
                "max_duration": 15,
                "supports_audio": True,
                "supports_negative_prompt": False,
                "supports_seed": True,
                "supports_upscale": True,
                "max_reference_images": 6,
            }

        def get_setup_schema(self) -> Dict[str, Any]:
            return {
                "name": DISPLAY,
                "badge": "free",
                "tag": "MiniMax H3 with sound on your own NVIDIA GPU through IV Studio + ComfyUI. Install IV Studio first.",
                "env_vars": [],
            }

        def generate(self, prompt: str, *, model: Optional[str] = None, image_url: Optional[str] = None,
                     reference_image_urls: Optional[list[str]] = None, duration: Optional[int] = None,
                     aspect_ratio: str = "16:9", resolution: str = "720p", negative_prompt: Optional[str] = None,
                     audio: Optional[bool] = None, seed: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
            model = model or "minimax-h3"
            def fail(msg: str, kind: str = "provider_error") -> Dict[str, Any]:
                return error_response(error=msg, error_type=kind, provider=NAME, model=model, prompt=prompt, aspect_ratio=aspect_ratio)

            if not (prompt or "").strip():
                return fail("prompt is required", "invalid_request")
            aspect = aspect_ratio if aspect_ratio in H3_ASPECTS else "16:9"
            seconds = max(1, min(15, int(duration or 5)))
            res = RESOLUTIONS.get(str(resolution or "720p").lower(), RESOLUTIONS["720p"])
            if kwargs.get("upscale"):
                res = RESOLUTIONS["1080p"]
            folder = _data_dir() / "inputs"
            folder.mkdir(exist_ok=True)
            started = time.monotonic()
            try:
                media: list[dict[str, str]] = []
                mode = "text"
                if image_url:
                    media = [{"kind": "picture", "file": _local_file(image_url, folder)}]
                    mode = "frames"
                elif reference_image_urls:
                    media = [{"kind": "picture", "file": _local_file(u, folder)} for u in reference_image_urls[:6]]
                    mode = "reference"
                with IVStudio(_exe_override()) as iv:
                    chars = iv.call("list_characters") or []
                    c = _pick_character(prompt, chars, kwargs.get("character"))
                    char_id = c["id"] if c else "none"
                    text = _with_char(prompt, c)
                    wrote = None
                    if not _H3_MARKERS.search(text) or (mode == "reference" and "<Picture" not in text):
                        # A plain idea: IV Studio's prompt writer (the one picked in its Settings) writes the H3 prompt.
                        try:
                            w = iv.call("write_prompt", {"idea": text, "mode": mode, "seconds": seconds, "aspect": aspect,
                                                         "character": char_id, **({"media": media} if media else {})}, timeout=300)
                            if isinstance(w, dict) and w.get("prompt"):
                                text, wrote = w["prompt"], w.get("engine")
                        except IVStudioError as e:
                            logger.info("iv-studio: prompt writer unavailable (%s); rendering the prompt as written", e)
                    ask = {"engine": "h3", "mode": mode, "prompt": text, "character": char_id, "aspect": aspect,
                           "seconds": seconds, "previews": 1, "projectName": f"Hermes · {prompt.strip()[:48]}"}
                    if seed is not None:
                        ask["seed"] = int(seed)
                    if media:
                        ask["media"] = media
                    try:
                        job = iv.call("generate_previews", ask, timeout=180)
                    except IVStudioError as e:
                        if "prompt checker" not in str(e):
                            raise
                        job = iv.call("generate_previews", {**ask, "force": True}, timeout=180)
                    job = _wait_job(iv, job["id"], "The take")
                    out = next((o for o in job.get("outputs", []) if o.get("kind") in ("preview", "candidate")), None) or (job.get("outputs") or [None])[0]
                    final_job = None
                    if res["finish"]:
                        fin = iv.call("finish_candidate", {"jobId": job["id"], "candidate": 1, **res["finish"]}, timeout=180)
                        final_job = _wait_job(iv, fin["id"], "The finish")
                        out = next((o for o in final_job.get("outputs", []) if o.get("kind") == "final"), None) or (final_job.get("outputs") or [out])[0]
                    if not out:
                        return fail("IV Studio finished but returned no video.")
                    video = _fetch_output(out, _data_dir(), "iv-studio")
                    try:
                        from agent.video_gen_provider import save_bytes_video
                        video = str(save_bytes_video(Path(video).read_bytes(), prefix="iv-studio", extension=Path(video).suffix.lstrip(".") or "mp4"))
                    except Exception:
                        pass
                    resolved = (final_job or job).get("plan", {}).get("resolved", {}) or job.get("plan", {}).get("resolved", {})
            except IVStudioError as e:
                return fail(str(e), "missing_dependency" if "isn't installed" in str(e) else "provider_error")
            except Exception as e:  # never take Hermes down with a render
                logger.debug("iv-studio video failed", exc_info=True)
                return fail(f"IV Studio: {e}")
            return success_response(
                video=video, model=model, prompt=prompt, modality="image" if image_url else "text",
                aspect_ratio=aspect, duration=seconds, provider=NAME,
                extra={
                    "character": c.get("name") if c else None,
                    "h3_prompt": text,
                    "prompt_writer": wrote,
                    "seed": resolved.get("seeds", [seed])[0] if isinstance(resolved.get("seeds"), list) else seed,
                    "upscaled": bool(res["finish"] and res["finish"].get("upscale2x")),
                    "finished": bool(res["finish"]),
                    "iv_studio_project": job.get("projectId"),
                    "render_seconds": round(time.monotonic() - started),
                    "cost": "free (local GPU)",
                },
            )

    return IVStudioVideo


# ---------------------------------------------------------------------------------------------------------------
# pictures

_IMAGE_ASPECT = {"landscape": "16:9", "portrait": "9:16", "square": "1:1"}
_GENERATE_ORDER = ("zimage", "krea2", "flux3", "qwen", "flux2", "sdxl", "moss")
_EDIT_ORDER = ("flux3", "qwen", "krea2", "flux2")


def _image_provider_class():
    from agent.image_gen_provider import ImageGenProvider, error_response, resolve_aspect_ratio, success_response

    class IVStudioImage(ImageGenProvider):
        """Pictures on the person's own GPU through IV Studio: the first ready family of Z-Image Turbo, Krea 2,
        Flux 3 Klein, Qwen Image 2.1, Flux.2 Klein, SDXL or Moss. Edits go to a family that edits."""

        @property
        def name(self) -> str:
            return NAME

        @property
        def display_name(self) -> str:
            return DISPLAY

        def is_available(self) -> bool:
            return find_command(_exe_override()) is not None

        def list_models(self) -> List[Dict[str, Any]]:
            return [{"id": "auto", "display": "Best ready model (local)", "speed": "about 15-60 s",
                     "strengths": "Whatever picture models IV Studio has set up", "price": "free (your own GPU)"}]

        def default_model(self) -> Optional[str]:
            return "auto"

        def capabilities(self) -> Dict[str, Any]:
            return {"max_reference_images": 5, "supports_edit": True}

        def get_setup_schema(self) -> Dict[str, Any]:
            return {"name": DISPLAY, "badge": "free", "tag": "Pictures on your own GPU through IV Studio + ComfyUI.", "env_vars": []}

        def generate(self, prompt: str, aspect_ratio: str = "landscape", *, image_url: Optional[str] = None,
                     reference_image_urls: Optional[list[str]] = None, **kwargs: Any) -> Dict[str, Any]:
            aspect_ratio = resolve_aspect_ratio(aspect_ratio)
            def fail(msg: str) -> Dict[str, Any]:
                return error_response(error=msg, error_type="provider_error", provider=NAME, model="auto", prompt=prompt, aspect_ratio=aspect_ratio)
            if not (prompt or "").strip():
                return fail("prompt is required")
            folder = _data_dir() / "inputs"
            folder.mkdir(exist_ok=True)
            try:
                refs = [_local_file(u, folder) for u in ([image_url] if image_url else []) + list(reference_image_urls or [])]
                edit = bool(refs)
                with IVStudio(_exe_override()) as iv:
                    fams = {f["id"]: f for f in (iv.call("list_image_models") or {}).get("families", [])}
                    order = _EDIT_ORDER if edit else _GENERATE_ORDER
                    fam = next((fams[i] for i in order if i in fams and fams[i].get("ready") and fams[i].get("models") and (not edit or fams[i].get("edit"))), None)
                    if not fam:
                        return fail("IV Studio has no picture model set up yet. Open IV Studio → Create → Image and click a model to download it.")
                    ask = {"mode": "edit" if edit else "generate", "model": fam["models"][0], "prompt": prompt,
                           "aspect": _IMAGE_ASPECT[aspect_ratio], "megapixels": 1, "count": 1, "name": f"Hermes · {prompt[:40]}"}
                    if edit:
                        ask["refs"] = refs[: int(fam.get("maxRefs") or 1)]
                    if kwargs.get("seed") is not None:
                        ask["seed"] = int(kwargs["seed"])
                    made = iv.call("make_image", ask, timeout=180)
                    deadline = time.monotonic() + 15 * 60
                    while made.get("status") not in ("success", "error"):
                        if time.monotonic() > deadline:
                            return fail("The picture is still being made; it keeps going in IV Studio.")
                        time.sleep(2.5)
                        made = iv.call("get_image", {"imageId": made["id"]}, timeout=60)
                    if made.get("status") == "error":
                        return fail(f"IV Studio: {made.get('error') or 'the picture failed'}")
                    files = made.get("files") or []
                    first = files[0] if files else None
                    path = first if isinstance(first, str) else (first or {}).get("path") or _fetch_output(first or {}, _data_dir(), "iv-image")
            except Exception as e:
                return fail(f"IV Studio: {e}")
            return success_response(image=str(path), model=fam["label"], prompt=prompt, aspect_ratio=aspect_ratio,
                                    provider=NAME, modality="image" if edit else "text",
                                    extra={"cost": "free (local GPU)", "iv_studio_model": fam["models"][0]})

    return IVStudioImage


# ---------------------------------------------------------------------------------------------------------------
# tool

STATUS_SCHEMA = {
    "name": "iv_studio_status",
    "description": (
        "Check IV Studio, the free local video and picture studio on this PC: whether it's installed, whether its "
        "ComfyUI and GPU are ready, the IV Studio characters (trained looks you can name in a video_generate prompt, "
        "like Tess), and anything it still needs. Call it before the first local video, or when a render fails."
    ),
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}


def _status_tool(args: dict, **kwargs: Any) -> str:
    try:
        return json.dumps(_status(_exe_override()))
    except Exception as e:
        return json.dumps({"installed": find_command(_exe_override()) is not None, "ready": False, "error": str(e)})


def register(ctx) -> None:
    """Hermes plugin entry point."""
    global _CTX
    _CTX = ctx
    ctx.register_tool(
        name="iv_studio_status", toolset="iv_studio", schema=STATUS_SCHEMA, handler=_status_tool,
        description=STATUS_SCHEMA["description"], emoji="🎬",
    )
    try:
        ctx.register_video_gen_provider(_video_provider_class()())
    except Exception as e:  # an older Hermes without video providers still gets the status tool
        logger.warning("iv-studio: video provider not registered: %s", e)
    try:
        ctx.register_image_gen_provider(_image_provider_class()())
    except Exception as e:
        logger.warning("iv-studio: image provider not registered: %s", e)
    skill = Path(__file__).parent / "skills" / "iv-studio-video" / "SKILL.md"
    if skill.is_file():
        try:
            ctx.register_skill("iv-studio-video", skill)
        except Exception as e:
            logger.debug("iv-studio: skill not registered: %s", e)
