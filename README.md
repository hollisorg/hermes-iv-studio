# IV Studio for Hermes

**Free, local video and pictures for Hermes, on your own NVIDIA GPU.**

This plugin makes [IV Studio](https://github.com/hollisorg/iv-studio) a backend for Hermes's own `video_generate`
and `image_generate` tools. Ask Hermes for a video in plain words, from the CLI, the Desktop app or Telegram, and it
renders on your PC with MiniMax H3 through ComfyUI: picture, sound and speech in one render, your own trained
characters by name, and no API key or credits.

```
you:    make a 5 second video of Tess at a rainy bus stop at night. She sighs and says "Of course it's late."
hermes: [video_generate → IV Studio on your RTX GPU, about 3 minutes]
        Here it is: 1376x768, 5 s, with her line. Seed 4242 if you want it again.
```

## What it does

- **`video_generate` on your GPU.** Text to video, a picture as the first frame (`image_url`), or up to six
  pictures as references (`reference_image_urls`). 1 to 15 seconds, 8 aspect ratios, sound and speech always on.
- **Plain words in, an H3 prompt out.** A plain idea goes through IV Studio's prompt writer (the one picked in its
  Settings: Claude Code, Codex, a local model through Ollama, or an OpenAI-compatible server) before it renders. A
  prompt already written for H3 renders as written.
- **Your characters by name.** IV Studio characters are trained LoRAs (IV Studio ships Tess and Theo, and you can
  train your own). Name one in the request ("Tess at a bus stop") and their trained look is used.
- **Finished, not just previewed.** `resolution` picks how far it goes: 480p is the raw take, 720p finishes it at
  about 1376x768 with DLSS 5, and 1080p finishes and upscales 2x to about 2752x1536.
- **`image_generate` on your GPU.** The first picture model IV Studio has ready (Z-Image Turbo, Krea 2, Flux 3 Klein,
  Qwen Image 2.1, Flux.2 Klein, SDXL or Moss); with `image_url` it edits that picture.
- **`iv_studio_status`.** A tool that tells Hermes whether IV Studio, ComfyUI and the GPU are ready, which characters
  exist and what's missing.
- **A bundled skill** (`iv-studio:iv-studio-video`) on writing good requests for it.

Everything you make is also in IV Studio's Library, where you can hunt more takes, finish at other sizes, or cut
renders together on its timeline.

## Needs

- Windows, Linux or a Mac with an NVIDIA GPU (16 GB or more is comfortable; IV Studio's speed presets cover smaller
  cards). A Mac renders only through IV Studio's cloud modes.
- [IV Studio](https://github.com/hollisorg/iv-studio/releases/latest) installed and opened once. Its Setup page
  installs ComfyUI's node packs and the MiniMax H3 models.
- Hermes 0.21.5 or newer.

## Install

```bash
hermes plugins install iv-studio
hermes config set video_gen.provider iv-studio
hermes config set image_gen.provider iv-studio   # optional: pictures too
```

Or pick **IV Studio (your GPU, free)** under `hermes tools` → Video Generation. If IV Studio is installed somewhere
unusual (a portable copy), set the plugin's `ivstudio_path` setting to its executable.

## How it works

IV Studio ships an MCP server inside its app. For each request the plugin finds the installed IV Studio
(`C:\Program Files\IV Studio`, `%LOCALAPPDATA%\Programs\IV Studio`, `/Applications/IV Studio.app`, or the
AppImage's starter script), starts that server as a child process for the length of the request, and talks to it
over stdio: it picks the character, has the prompt written, starts one take with `generate_previews`, waits for it,
finishes it with `finish_candidate`, and copies the result into Hermes's media cache. The GPU work happens in your
own ComfyUI, exactly as when you click Hunt in IV Studio's window.

## Disclosure

- **Runs a local program:** IV Studio's own executable, in Node mode on its bundled MCP server, as a child process
  for each request; it stops when the request ends.
- **Network:** none of its own. It reads finished files from your local ComfyUI (`127.0.0.1`), and downloads a
  picture only when you pass an `http(s)` URL as `image_url` or a reference. IV Studio's prompt writer uses whatever
  writer you picked in IV Studio's Settings (a local model, or Claude Code / Codex on their own logins).
- **Files:** pictures you pass by URL are saved under `HERMES_HOME/plugin-data/iv-studio/inputs`; finished videos are
  copied into Hermes's media cache. Renders also stay in IV Studio's Library and ComfyUI's output folder.
- **Credentials:** none. No API keys, no telemetry, no self-update.
- **Terms:** MiniMax H3 runs under the MiniMax H3 Community License; IV Studio asks you to agree to its terms once
  before your first render (real people only with their consent, nothing involving minors, say it's AI when you post
  it). Its territory clause excludes the EU, UK, South Korea and the USA unless MiniMax licenses you.

## Licence

MIT. IV Studio itself is MIT too: https://github.com/hollisorg/iv-studio
