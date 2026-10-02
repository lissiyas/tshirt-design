import asyncio
import base64
import hashlib
import hmac
import io
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
IMG_DIR = DATA_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
SECRET = os.getenv("SESSION_SECRET") or hashlib.sha256((APP_PASSWORD + "imagen").encode()).hexdigest()
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "600"))

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
MODELS = {
    "gpt-image-2":   {"label": "GPT Image 2 (native up to 4K)", "customSize": True},
    "gpt-image-1.5": {"label": "GPT Image 1.5 (upscaled)", "customSize": False},
    "gpt-image-1":   {"label": "GPT Image 1 (upscaled)", "customSize": False},
}
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL") if os.getenv("DEFAULT_MODEL") in MODELS else "gpt-image-2"

app = FastAPI(title="Imagen Studio")

# ---------- auth (optional password, cookie-based) ----------
def _token() -> str:
    return hmac.new(SECRET.encode(), b"ok", hashlib.sha256).hexdigest()


def _authed(request: Request) -> bool:
    if not APP_PASSWORD:
        return True
    return hmac.compare_digest(request.cookies.get("auth", ""), _token())


def require_auth(request: Request):
    if not _authed(request):
        raise HTTPException(401, "Not authorized")


class LoginBody(BaseModel):
    password: str = ""


@app.post("/api/login")
def login(body: LoginBody, request: Request, response: Response):
    if not APP_PASSWORD:
        return {"ok": True}
    if hmac.compare_digest(body.password.encode(), APP_PASSWORD.encode()):
        secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
        response.set_cookie("auth", _token(), httponly=True, samesite="lax", secure=secure, max_age=60 * 60 * 24 * 30)
        return {"ok": True}
    raise HTTPException(401, "Wrong password")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/config")
def config(request: Request):
    return {
        "needsPassword": bool(APP_PASSWORD),
        "authed": _authed(request),
        "keyConfigured": bool(OPENAI_KEY),
        "models": [{"id": k, **v} for k, v in MODELS.items()],
        "defaultModel": DEFAULT_MODEL,
        "ratios": list(SIZES.keys()),
        "sizes": SIZES,
    }


# ---------- generation ----------
MAX_REF_IMAGES = 5
MAX_REF_BYTES = 20 * 1024 * 1024
REF_TYPES = {"image/png", "image/jpeg", "image/webp"}

JOBS: dict[str, dict] = {}


def _native_fallback(w: int, h: int) -> str:
    if w > h:
        return "1536x1024"
    if h > w:
        return "1024x1536"
    return "1024x1024"


def _process_image(raw: bytes, target: tuple[int, int], upscale: bool, fmt: str) -> tuple[bytes, int, int, bool]:
    img = Image.open(io.BytesIO(raw))
    upscaled = False
    if upscale:
        tw, th = target
        # cover-crop to target aspect, then Lanczos resize
        scale = max(tw / img.width, th / img.height)
        nw, nh = round(img.width * scale), round(img.height * scale)
        img = img.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - tw) // 2, (nh - th) // 2
        img = img.crop((left, top, left + tw, top + th))
        upscaled = True
    buf = io.BytesIO()
    if fmt == "jpeg":
        img.convert("RGB").save(buf, "JPEG", quality=95, subsampling=0)
    elif fmt == "webp":
        img.save(buf, "WEBP", quality=95)
    else:
        img.save(buf, "PNG")
    return buf.getvalue(), img.width, img.height, upscaled


async def _run_job(job: dict):
    p = job["params"]
    tw, th = SIZES[p["ratio"]][p["resolution"]]
    custom = MODELS[p["model"]]["customSize"]
    size = f"{tw}x{th}" if custom else _native_fallback(tw, th)
    body = {
        "model": p["model"], "prompt": p["prompt"], "size": size,
        "n": p["n"], "quality": p["quality"], "output_format": p["format"],
    }
    refs = job.get("refs") or []
    headers = {"Authorization": f"Bearer {OPENAI_KEY}"}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=15)) as client:
            if refs:  # reference images attached -> image edit endpoint (multipart)
                r = await client.post(
                    f"{OPENAI_BASE}/images/edits",
                    headers=headers,
                    data={k: str(v) for k, v in body.items()},
                    files=[("image[]", (name, content, mime)) for name, content, mime in refs],
                )
            else:
                r = await client.post(f"{OPENAI_BASE}/images/generations", headers=headers, json=body)
        try:
            data = r.json()
        except Exception:
            data = {}
        if r.status_code >= 400:
            raise RuntimeError((data.get("error") or {}).get("message") or f"OpenAI returned {r.status_code}")

        out = []
        for item in data.get("data", []):
            b64 = item.get("b64_json")
            if not b64:
                continue
            needs_upscale = (not custom) and (tw > 1536 or th > 1536)
            img_bytes, w, h, upscaled = await asyncio.to_thread(
                _process_image, base64.b64decode(b64), (tw, th), needs_upscale, p["format"]
            )
            iid = str(uuid.uuid4())
            ext = "jpg" if p["format"] == "jpeg" else p["format"]
            (IMG_DIR / f"{iid}.{ext}").write_bytes(img_bytes)
            rec = {
                "id": iid, "file": f"{iid}.{ext}", "prompt": p["prompt"], "model": p["model"],
                "quality": p["quality"], "ratio": p["ratio"], "requestedSize": f"{tw}x{th}",
                "width": w, "height": h, "upscaled": upscaled, "bytes": len(img_bytes),
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "revisedPrompt": item.get("revised_prompt"),
                "edited": bool(refs),
            }
            (IMG_DIR / f"{iid}.json").write_text(json.dumps(rec))
            out.append(rec)
        if not out:
            raise RuntimeError("No image returned")
        job["status"], job["images"] = "done", out
    except httpx.TimeoutException:
        job["status"], job["error"] = "error", "Timed out waiting for OpenAI"
    except Exception as e:  # noqa: BLE001
        job["status"], job["error"] = "error", str(e)
    finally:
        job["finishedAt"] = time.time()


def _prune_jobs():
    cutoff = time.time() - 3600
    for k in [k for k, j in JOBS.items() if j["createdAt"] < cutoff]:
        JOBS.pop(k, None)


@app.post("/api/generate", dependencies=[Depends(require_auth)])
async def generate(
    prompt: str = Form(...),
    model: str = Form(DEFAULT_MODEL),
    ratio: str = Form("1:1"),
    resolution: str = Form("2K"),
    quality: str = Form("high"),
    format: str = Form("png"),
    n: int = Form(1),
    images: list[UploadFile] = File(default=[]),
):
    if not OPENAI_KEY:
        raise HTTPException(500, "OPENAI_API_KEY is not set on the server")
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(400, "Prompt is required")
    if len(prompt) > 32000:
        raise HTTPException(400, "Prompt too long")
    if len(images) > MAX_REF_IMAGES:
        raise HTTPException(400, f"At most {MAX_REF_IMAGES} reference images")
    refs = []
    for i, up in enumerate(images):
        content = await up.read()
        mime = up.content_type or ""
        if mime not in REF_TYPES:
            raise HTTPException(400, "Reference images must be PNG, JPEG or WebP")
        if len(content) > MAX_REF_BYTES:
            raise HTTPException(400, "Each reference image must be under 20 MB")
        ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}[mime]
        refs.append((f"ref{i}.{ext}", content, mime))
    _prune_jobs()
    params = {
        "prompt": prompt,
        "model": model if model in MODELS else DEFAULT_MODEL,
        "ratio": ratio if ratio in SIZES else "1:1",
        "resolution": resolution if resolution in ("1K", "2K", "4K") else "2K",
        "quality": quality if quality in ("low", "medium", "high", "auto") else "high",
        "format": format if format in ("png", "jpeg", "webp") else "png",
        "n": min(max(n, 1), 4),
    }
    jid = str(uuid.uuid4())
    job = {"id": jid, "status": "running", "createdAt": time.time(), "params": params, "refs": refs}
    JOBS[jid] = job
    asyncio.create_task(_run_job(job))
    return {"jobId": jid}


@app.get("/api/jobs/{jid}", dependencies=[Depends(require_auth)])
def job_status(jid: str):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "Unknown job")
    return {
        "status": j["status"], "images": j.get("images"), "error": j.get("error"),
        "elapsedMs": int(((j.get("finishedAt") or time.time()) - j["createdAt"]) * 1000),
    }


@app.get("/api/gallery", dependencies=[Depends(require_auth)])
def gallery():
    items = []
    for f in IMG_DIR.glob("*.json"):
        try:
            items.append(json.loads(f.read_text()))
        except Exception:
            pass
    items.sort(key=lambda x: x["createdAt"], reverse=True)
    return items[:200]


_FILE_RE = re.compile(r"^[\w-]+\.(png|jpg|webp)$")


@app.get("/images/{name}", dependencies=[Depends(require_auth)])
def get_image(name: str, download: int = 0):
    if not _FILE_RE.match(name) or not (IMG_DIR / name).exists():
        raise HTTPException(404, "Not found")
    headers = {"Cache-Control": "private, max-age=31536000, immutable"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{name}"'
    return FileResponse(IMG_DIR / name, headers=headers)


@app.delete("/api/images/{iid}", dependencies=[Depends(require_auth)])
def delete_image(iid: str):
    if not re.match(r"^[\w-]+$", iid):
        raise HTTPException(400, "Bad id")
    meta = IMG_DIR / f"{iid}.json"
    if not meta.exists():
        raise HTTPException(404, "Not found")
    rec = json.loads(meta.read_text())
    (IMG_DIR / rec["file"]).unlink(missing_ok=True)
    meta.unlink(missing_ok=True)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.on_event("startup")
def _warn():
    if not OPENAI_KEY:
        print("WARNING: OPENAI_API_KEY is not set")
    if not APP_PASSWORD:
        print("WARNING: APP_PASSWORD is not set; anyone with the URL can spend your credits")
