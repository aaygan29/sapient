#!/usr/bin/env bash
# Share the Sapient sandbox with a public URL — no accounts, no deploy.
# Run this in a terminal and KEEP IT OPEN while sharing. Ctrl+C to stop.
#
#   ./share.sh
#
# It prints a public https://<random>.trycloudflare.com URL — send <URL>/demo
# to an investor along with their API key (shown below / mint with issue_key.py).
set -e
cd "$(dirname "$0")"
PORT="${PORT:-8000}"

# 1) environment
if [ ! -d .venv ]; then python3.12 -m venv .venv; fi
. .venv/bin/activate
pip install -q -e . >/dev/null 2>&1 || true

# 2) ensure at least one API key exists
if [ ! -s keys.json ]; then
  echo ">> No keys yet — issuing one for 'investor-demo':"
  python scripts/issue_key.py --client-id investor-demo
  echo ">> (mint more anytime: python scripts/issue_key.py --client-id NAME)"
fi

# 3) start the sandbox (mock engine — secure, no weights)
SAPIENT_ENGINE=mock SAPIENT_KEYS_FILE=keys.json \
  python -m uvicorn sapient_serving.api.app:app --host 127.0.0.1 --port "$PORT" \
  >/tmp/sapient_uvicorn.log 2>&1 &
UV=$!
trap 'kill "$UV" 2>/dev/null' EXIT
for _ in $(seq 1 60); do curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break; sleep 0.5; done
echo ">> Sandbox up on 127.0.0.1:$PORT"

# 4) public tunnel (URL prints below; share it as  <URL>/demo)
echo ">> Opening public tunnel — share the https://...trycloudflare.com URL with /demo appended:"
exec cloudflared tunnel --url "http://127.0.0.1:$PORT"
