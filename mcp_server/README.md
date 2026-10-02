# OpenAI Image MCP server

Gives Claude (or any MCP client) three tools:

| Tool | What it does |
|---|---|
| `generate_image` | Text → image. `ratio` (1:1, 16:9, 9:16, 3:2, 2:3, 4:3, 3:4), `resolution` (1K/2K/4K), `quality`, `n` (1-4), `output_format`, `model` |
| `edit_image` | Edit/remix 1-5 local images with a prompt |
| `list_generated_images` | Recent images with paths and prompts |

Full-resolution files are saved to `IMAGE_OUTPUT_DIR` (default `~/imagen-mcp-images`); the tool returns the path plus a
1024px JPEG preview (so a 4K image doesn't flood the chat). `gpt-image-2` is asked for native sizes up to 3840px;
older models are generated at ≤1536px and upscaled (reported as "upscaled").

## Install
```bash
cd mcp_server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Use locally (stdio) — Claude Desktop / Claude Code
Claude Code:
```bash
claude mcp add openai-image -e OPENAI_API_KEY=sk-... -- /full/path/to/.venv/bin/python /full/path/to/mcp_server/server.py
```
Claude Desktop (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "openai-image": {
      "command": "/full/path/to/.venv/bin/python",
      "args": ["/full/path/to/mcp_server/server.py"],
      "env": { "OPENAI_API_KEY": "sk-...", "IMAGE_OUTPUT_DIR": "/Users/you/Pictures/ai" }
    }
  }
}
```

## Use remotely (streamable HTTP)
```bash
MCP_TRANSPORT=streamable-http MCP_HOST=0.0.0.0 MCP_PORT=8001 \
OPENAI_API_KEY=sk-... MCP_AUTH_TOKEN=long-random-string python server.py
# endpoint: http://<host>:8001/mcp   (send  Authorization: Bearer <token>)
```
Always set `MCP_AUTH_TOKEN` and put it behind HTTPS. Note: with a remote server, saved file paths live on the server,
so `edit_image` can only use files that exist there; the preview is what travels back to the client.
