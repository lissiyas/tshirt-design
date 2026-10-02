"""MCP server for OpenAI image models (generate + edit, up to 4K).

Transports:
  stdio (default)        -> local use from Claude Desktop / Claude Code
  streamable-http        -> remote use; set MCP_TRANSPORT=streamable-http (protect with MCP_AUTH_TOKEN)

Env:
  OPENAI_API_KEY       required
  IMAGE_OUTPUT_DIR     where full-resolution files are saved (default ~/imagen-mcp-images)
  DEFAULT_MODEL        default gpt-image-2
  OPENAI_BASE_URL      default https://api.openai.com/v1
  MCP_TRANSPORT        stdio | streamable-http
  MCP_HOST / MCP_PORT  for HTTP (default 127.0.0.1:8001)
  MCP_AUTH_TOKEN       if set, HTTP requests must send "Authorization: Bearer <token>"
"""
import asyncio
import base64
import hmac
import io
import json
import mimetypes
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
from mcp.server.mcpserver import Context, Image, MCPServer
from PIL import Image as PILImage

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OUT_DIR = Path(os.getenv("IMAGE_OUTPUT_DIR", Path.home() / "imagen-mcp-images")).expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "600"))

Ratio = Literal["1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4"]
Resolution = Literal["1K", "2K", "4K"]
Quality = Literal["low", "medium", "high", "auto"]
Fmt = Literal["png", "jpeg", "webp"]

# gpt-image-2 accepts custom sizes (edges multiple of 16, max edge 3840, ~8.3MP cap, ratio <= 3:1).
SIZES = {
    "1:1":  {"1K": (1024, 1024), "2K": (2048, 2048), "4K": (2880, 2880)},
    "16:9": {"1K": (1280, 720),  "2K": (2560, 1440), "4K": (3840, 2160)},
    "9:16": {"1K": (720, 1280),  "2K": (1440, 2560), "4K": (2160, 3840)},
    "3:2":  {"1K": (1536, 1024), "2K": (3072, 2048), "4K": (3504, 2336)},
    "2:3":  {"1K": (1024, 1536), "2K": (2048, 3072), "4K": (2336, 3504)},
    "4:3":  {"1K": (1024, 768),  "2K": (2048, 1536), "4K": (3264, 2448)},
    "3:4":  {"1K": (768, 1024),  "2K": (1536, 2048), "4K": (2448, 3264)},
}
CUSTOM_SIZE_MODELS = {"gpt-image-2"}
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-image-2")

mcp = MCPServer(
    "openai-image",
    instructions=(
        "Generate and edit images with OpenAI image models. Full-resolution files are saved to disk and the path is "
        "returned; a small preview is attached. 4K generations can take 1-3 minutes."
    ),
)


def _native_fallback(w: int, h: int) -> str:
    return "1536x1024" if w > h else "1024x1536" if h > w else "1024x1024"


def _finalize(raw: bytes, target: tuple[int, int], upscale: bool, fmt: str) -> tuple[bytes, int, int, bool]:
    img = PILImage.open(io.BytesIO(raw))
    if upscale:
        tw, th = target
        s = max(tw / img.width, th / img.height)
        nw, nh = round(img.width * s), round(img.height * s)
        img = img.resize((nw, nh), PILImage.LANCZOS)
        l, t = (nw - tw) // 2, (nh - th) // 2
        img = img.crop((l, t, l + tw, t + th))
    buf = io.BytesIO()
    if fmt == "jpeg":
        img.convert("RGB").save(buf, "JPEG", quality=95, subsampling=0)
    elif fmt == "webp":
        img.save(buf, "WEBP", quality=95)
    else:
        img.save(buf, "PNG")
    return buf.getvalue(), img.width, img.height, upscale


def _preview(raw: bytes, max_edge: int = 1024) -> bytes:
    img = PILImage.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((max_edge, max_edge), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    return buf.getvalue()


async def _post(ctx: Context | None, endpoint: str, **kwargs) -> dict:
    """POST to OpenAI, sending progress pings so clients don't time out on long 4K jobs."""
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set for this MCP server")

    async def call():
        async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=15)) as client:
            return await client.post(
                f"{OPENAI_BASE}/{endpoint}", headers={"Authorization": f"Bearer {OPENAI_KEY}"}, **kwargs
            )

    task = asyncio.create_task(call())
    waited = 0
    while not task.done():
        await asyncio.wait({task}, timeout=10)
        waited += 10
        if not task.done() and ctx is not None:
            try:
                await ctx.report_progress(waited, None, f"Generating… {waited}s")
            except Exception:
                pass
    r = task.result()
    try:
        data = r.json()
    except Exception:
        data = {}
    if r.status_code >= 400:
        raise RuntimeError((data.get("error") or {}).get("message") or f"OpenAI returned {r.status_code}")
    return data


async def _save_results(data: dict, prompt: str, model: str, ratio: str, target: tuple[int, int],
                        custom: bool, fmt: str, quality: str, return_preview: bool):
    out_content, lines = [], []
    for item in data.get("data", []):
        b64 = item.get("b64_json")
        if not b64:
            continue
        need_up = (not custom) and (target[0] > 1536 or target[1] > 1536)
        raw, w, h, up = await asyncio.to_thread(_finalize, base64.b64decode(b64), target, need_up, fmt)
        iid = uuid.uuid4().hex[:12]
        ext = "jpg" if fmt == "jpeg" else fmt
        path = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{iid}.{ext}"
        path.write_bytes(raw)
        path.with_suffix(".json").write_text(json.dumps({
            "prompt": prompt, "model": model, "ratio": ratio, "quality": quality, "width": w, "height": h,
            "upscaled": up, "createdAt": datetime.now(timezone.utc).isoformat(),
        }))
        lines.append(f"{path} ({w}x{h}, {len(raw)/1048576:.1f} MB{', upscaled' if up else ''})")
        if return_preview:
            out_content.append(Image(data=await asyncio.to_thread(_preview, raw), format="jpeg"))
    if not lines:
        raise RuntimeError("OpenAI returned no images")
    return [*out_content, "Saved:\n" + "\n".join(lines)]


@mcp.tool()
async def generate_image(
    prompt: str,
    ratio: Ratio = "16:9",
    resolution: Resolution = "4K",
    quality: Quality = "high",
    n: int = 1,
    output_format: Fmt = "png",
    model: str = DEFAULT_MODEL,
    return_preview: bool = True,
    ctx: Context | None = None,
):
    """Generate image(s) from a text prompt with an OpenAI image model.

    resolution 4K means ~3840px on the long edge (gpt-image-2 generates this natively; older models are
    generated at <=1536px and upscaled). Saves full-res files to disk and returns their paths plus a preview.
    """
    target = SIZES[ratio][resolution]
    custom = model in CUSTOM_SIZE_MODELS
    size = f"{target[0]}x{target[1]}" if custom else _native_fallback(*target)
    data = await _post(ctx, "images/generations", json={
        "model": model, "prompt": prompt, "size": size, "n": max(1, min(n, 4)),
        "quality": quality, "output_format": output_format,
    })
    return await _save_results(data, prompt, model, ratio, target, custom, output_format, quality, return_preview)


@mcp.tool()
async def edit_image(
    prompt: str,
    image_paths: list[str],
    ratio: Ratio = "1:1",
    resolution: Resolution = "2K",
    quality: Quality = "high",
    output_format: Fmt = "png",
    model: str = DEFAULT_MODEL,
    return_preview: bool = True,
    ctx: Context | None = None,
):
    """Edit or remix existing image(s) (absolute local file paths, up to 5) following the prompt."""
    if not image_paths or len(image_paths) > 5:
        raise ValueError("Provide 1-5 image paths")
    files = []
    for p in image_paths:
        fp = Path(p).expanduser()
        if not fp.is_file():
            raise FileNotFoundError(f"Not found: {p}")
        files.append(("image[]", (fp.name, fp.read_bytes(), mimetypes.guess_type(fp.name)[0] or "image/png")))
    target = SIZES[ratio][resolution]
    custom = model in CUSTOM_SIZE_MODELS
    size = f"{target[0]}x{target[1]}" if custom else _native_fallback(*target)
    data = await _post(ctx, "images/edits", data={
        "model": model, "prompt": prompt, "size": size, "quality": quality, "output_format": output_format,
    }, files=files)
    return await _save_results(data, prompt, model, ratio, target, custom, output_format, quality, return_preview)


@mcp.tool()
def list_generated_images(limit: int = 20) -> str:
    """List the most recently generated images (path, size, prompt)."""
    metas = sorted(OUT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[: max(1, min(limit, 100))]
    rows = []
    for m in metas:
        try:
            d = json.loads(m.read_text())
        except Exception:
            continue
        img = next((m.with_suffix(e) for e in (".png", ".jpg", ".webp") if m.with_suffix(e).exists()), None)
        if img:
            rows.append(f"{img} — {d['width']}x{d['height']} — {d['model']} — {d['prompt'][:100]}")
    return "\n".join(rows) or "No images yet."


def _http_app():
    """ASGI app for remote use, with optional bearer-token protection."""
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    app = mcp.streamable_http_app()
    token = os.getenv("MCP_AUTH_TOKEN", "")
    if token:
        class Auth(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                got = request.headers.get("authorization", "")
                if not hmac.compare_digest(got, f"Bearer {token}"):
                    return JSONResponse({"error": "unauthorized"}, status_code=401)
                return await call_next(request)
        app.add_middleware(Auth)
    return app


if __name__ == "__main__":
    if os.getenv("MCP_TRANSPORT", "stdio") == "streamable-http":
        import uvicorn
        if not os.getenv("MCP_AUTH_TOKEN"):
            print("WARNING: MCP_AUTH_TOKEN not set; anyone who can reach this port can spend your credits")
        uvicorn.run(_http_app(), host=os.getenv("MCP_HOST", "127.0.0.1"), port=int(os.getenv("MCP_PORT", "8001")))
    else:
        mcp.run()
