#!/usr/bin/env bash
# app.sh — starts the full RAG application stack:
#   1. Ollama (if not already running)
#   2. FastAPI backend  (port 8000)
#   3. React frontend   (port 5173)
#
# Usage:
#   chmod +x app.sh   (first time only)
#   ./app.sh
#
# Press Ctrl+C once to stop all three processes.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── Colours ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[app]${NC} $*"; }
warning() { echo -e "${YELLOW}[app]${NC} $*"; }
error()   { echo -e "${RED}[app]${NC} $*"; }

# ── Cleanup: kill child processes on exit ──────────────────────────────────────
PIDS=()
cleanup() {
  echo ""
  info "Shutting down…"
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null
  info "Done."
}
trap cleanup EXIT INT TERM

# ── Logs directory
LOGS_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOGS_DIR"

# ── 1. Ollama ──────────────────────────────────────────────────────────────────
if pgrep -x ollama > /dev/null; then
  info "Ollama is already running — skipping."
else
  if ! command -v ollama &> /dev/null; then
    error "Ollama is not installed. Install it from https://ollama.com and re-run."
    exit 1
  fi
  info "Starting Ollama…"
  ollama serve &> "$LOGS_DIR/ollama.log" &
  PIDS+=($!)
  sleep 2   # give Ollama a moment to bind its port
  info "Ollama started (logs → logs/ollama.log)"
fi

# ── 2. FastAPI backend ─────────────────────────────────────────────────────────
info "Starting FastAPI backend on http://localhost:8000 …"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
  warning "No .venv found — make sure your virtual environment is activated."
fi

uvicorn backend.api.main:app --port 8000 &> "$LOGS_DIR/backend.log" &
PIDS+=($!)
info "Backend started (logs → logs/backend.log)"

# ── 3. React frontend ──────────────────────────────────────────────────────────
info "Starting React frontend on http://localhost:5173 …"
cd "$PROJECT_ROOT/frontend"
npm run dev &> "$LOGS_DIR/frontend.log" &
PIDS+=($!)
info "Frontend started (logs → logs/frontend.log)"

# ── Ready ──────────────────────────────────────────────────────────────────────
echo ""
info "Stack is up:"
echo -e "  Frontend  → ${GREEN}http://localhost:5173${NC}"
echo -e "  Backend   → ${GREEN}http://localhost:8000${NC}"
echo -e "  API docs  → ${GREEN}http://localhost:8000/docs${NC}"
echo ""
info "Press Ctrl+C to stop everything."

# Keep the script alive until the user interrupts
wait
