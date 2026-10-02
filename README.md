# Imagen Studio

FastAPI backend + chat-style web UI for generating high-resolution images with OpenAI's image models.
Type a prompt, or attach up to 5 reference images (button, paste, or drag-and-drop) and describe the change; results
come back in the thread with Download, Edit (reuse as a reference) and Regenerate. Chats are kept in your browser;
images are kept on the server.
Your API key stays on the server; the browser never sees it.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
export APP_PASSWORD=choose-a-password
uvicorn app.main:app --reload
# open http://localhost:8000
```

## Deploy (Docker)
```bash
docker build -t imagen-studio .
docker run -p 8000:8000 -v imagen-data:/data \
  -e OPENAI_API_KEY=sk-... -e APP_PASSWORD=... imagen-studio
```
Works on Render, Railway, Fly.io, or any VPS. Set `OPENAI_API_KEY` and `APP_PASSWORD` as environment
variables and mount a persistent volume at `/data` so the gallery survives restarts.
Put it behind HTTPS (the platforms above do this for you).

## Notes
- **4K:** `gpt-image-2` is requested at the exact size (e.g. 3840x2160). Older models (`gpt-image-1.5`, `gpt-image-1`)
  only generate up to 1536px, so the app upscales them with Lanczos; the gallery marks those as "upscaled".
- **Sizes** live in the `SIZES` table in `app/main.py`. If OpenAI rejects a size, adjust it there.
- **Timeouts:** 4K can take 1-3 minutes, so generation runs as a background job that the UI polls. If your host
  has a short request timeout this is fine, since each poll is quick.
- **Cost:** 4K high-quality images use many tokens. Always set `APP_PASSWORD` on a public deployment.
- Jobs are held in memory, so run a single worker (the default) or move jobs to Redis if you scale out.
