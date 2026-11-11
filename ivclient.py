"""A small MCP client for IV Studio's own MCP server.

IV Studio (https://github.com/hollisorg/iv-studio) ships an MCP server inside its app: its executable run in Node
mode on the bundled ``mcp.js``. This module finds an installed IV Studio, starts that server as a child process,
speaks MCP's JSON-RPC over stdio (one JSON message per line) and stops it again. Standard library only.

Nothing here touches the network: the child process talks to the person's own ComfyUI the way the IV Studio window
does.
"""

from __future__ import annotations

import json
import os
import platform
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

PROTOCOL = "2025-06-18"
CLIENT = {"name": "hermes-iv-studio", "version": "1.0.0"}


class IVStudioError(RuntimeError):
    """IV Studio is missing, its server failed, or a tool returned an error."""


def _candidates() -> list[tuple[Path, Path]]:
    """(executable, mcp.js) pairs where IV Studio installs itself, most likely first."""
    out: list[tuple[Path, Path]] = []
    system = platform.system()
    if system == "Windows":
        roots = [
            Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "IV Studio",
            Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "Programs" / "IV Studio",
        ]
        for r in roots:
            out.append((r / "IV Studio.exe", r / "resources" / "app.asar" / "out" / "main" / "mcp.js"))
    elif system == "Darwin":
        for apps in (Path("/Applications"), Path.home() / "Applications"):
            app = apps / "IV Studio.app" / "Contents"
            out.append((app / "MacOS" / "IV Studio", app / "Resources" / "app.asar" / "out" / "main" / "mcp.js"))
    return out


def _linux_starter() -> Optional[Path]:
    """The AppImage build writes a starter script into its data folder each time it opens."""
    base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    for name in ("IV Studio", "iv-studio"):
        p = base / name / "mcp-start.sh"
        if p.is_file():
            return p
    return None


def find_command(exe_override: str = "") -> Optional[dict[str, Any]]:
    """How to start IV Studio's MCP server on this machine, or None when IV Studio isn't installed.

    ``exe_override`` is the IV Studio executable from the plugin's settings, for an install somewhere unusual (a
    portable copy, for example)."""
    if exe_override:
        exe = Path(exe_override)
        if exe.is_dir():
            exe = exe / ("IV Studio.exe" if platform.system() == "Windows" else "iv-studio")
        res = exe.parent / "resources" if platform.system() != "Darwin" else exe.parent.parent / "Resources"
        script = res / "app.asar" / "out" / "main" / "mcp.js"
        if exe.is_file():
            return {"command": str(exe), "args": [str(script)], "env": {"ELECTRON_RUN_AS_NODE": "1"}}
    for exe, script in _candidates():
        if exe.is_file():
            return {"command": str(exe), "args": [str(script)], "env": {"ELECTRON_RUN_AS_NODE": "1"}}
    if platform.system() == "Linux":
        starter = _linux_starter()
        if starter:
            return {"command": "/bin/sh", "args": [str(starter)], "env": {}}
    return None


class IVStudio:
    """One IV Studio MCP server for the length of a ``with`` block.

    >>> with IVStudio() as iv:
    ...     iv.call("list_characters")
    """

    def __init__(self, exe_override: str = "", start_timeout: float = 60.0):
        cmd = find_command(exe_override)
        if not cmd:
            raise IVStudioError(
                "IV Studio isn't installed on this computer. Get it free from "
                "https://github.com/hollisorg/iv-studio/releases/latest and open it once."
            )
        self._cmd = cmd
        self._start_timeout = start_timeout
        self._proc: Optional[subprocess.Popen] = None
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._next_id = 1
        self._stderr: list[str] = []

    # -- process ---------------------------------------------------------------------------------------------
    def __enter__(self) -> "IVStudio":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def start(self) -> None:
        env = {**os.environ, **self._cmd["env"], "IV_STUDIO_AGENT": "Hermes"}
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(
            [self._cmd["command"], *self._cmd["args"]],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, creationflags=flags, bufsize=0,
        )
        threading.Thread(target=self._read_out, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()
        self._request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {}, "clientInfo": CLIENT}, self._start_timeout)
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def close(self) -> None:
        p, self._proc = self._proc, None
        if not p:
            return
        try:
            if p.stdin:
                p.stdin.close()
            p.wait(timeout=5)
        except Exception:
            p.kill()

    def _read_out(self) -> None:
        assert self._proc and self._proc.stdout
        for raw in iter(self._proc.stdout.readline, b""):
            self._lines.put(raw.decode("utf-8", "replace"))
        self._lines.put(None)

    def _read_err(self) -> None:
        assert self._proc and self._proc.stderr
        for raw in iter(self._proc.stderr.readline, b""):
            self._stderr.append(raw.decode("utf-8", "replace").rstrip())
            del self._stderr[:-40]

    # -- JSON-RPC ---------------------------------------------------------------------------------------------
    def _send(self, msg: dict[str, Any]) -> None:
        if not self._proc or not self._proc.stdin:
            raise IVStudioError("IV Studio's server isn't running.")
        self._proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
        self._proc.stdin.flush()

    def _request(self, method: str, params: dict[str, Any], timeout: float) -> Any:
        rid = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise IVStudioError(f"IV Studio didn't answer {method} within {int(timeout)} s.")
            try:
                line = self._lines.get(timeout=min(left, 1.0))
            except queue.Empty:
                continue
            if line is None:
                tail = " | ".join(self._stderr[-3:])
                raise IVStudioError(f"IV Studio's server stopped. {tail}".strip())
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("id") != rid:
                continue  # notifications, logs
            if "error" in msg:
                raise IVStudioError(str(msg["error"].get("message") or msg["error"]))
            return msg.get("result")

    def call(self, tool: str, args: Optional[dict[str, Any]] = None, timeout: float = 120.0) -> Any:
        """Call an IV Studio tool and return its result (structured when the tool gives one, else parsed text)."""
        res = self._request("tools/call", {"name": tool, "arguments": args or {}}, timeout) or {}
        text = "\n".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text").strip()
        if res.get("isError"):
            raise IVStudioError(text or f"{tool} failed")
        if "structuredContent" in res and res["structuredContent"] is not None:
            sc = res["structuredContent"]
            return sc.get("items", sc) if isinstance(sc, dict) and set(sc) == {"items"} else sc
        try:
            return json.loads(text)
        except ValueError:
            return text
