#!/usr/bin/env bash
# scripts/setup.sh — Bootstrap the AlgoDollar development environment
set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
log_success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
log_error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }

# ── Header ────────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}AlgoDollar — Development Setup${RESET}\n"

# ── Dependency checks ─────────────────────────────────────────────────────────
MISSING=()

check_cmd() {
  local cmd="$1"
  local label="${2:-$1}"
  if command -v "$cmd" &>/dev/null; then
    local version
    version=$(command "$cmd" --version 2>&1 | head -n1)
    log_success "$label found: $version"
  else
    log_error "$label not found."
    MISSING+=("$label")
  fi
}

log_info "Checking required dependencies..."
check_cmd python3   "Python 3"
check_cmd node      "Node.js"
check_cmd docker    "Docker"
check_cmd docker    "Docker Compose (plugin)"
check_cmd gh        "GitHub CLI (gh)"

if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo ""
  log_error "The following required tools are missing:"
  for tool in "${MISSING[@]}"; do
    echo -e "  ${RED}•${RESET} $tool"
  done
  echo ""
  echo "Please install them and re-run this script."
  exit 1
fi

# Verify Docker Compose v2 plugin
if ! docker compose version &>/dev/null; then
  log_error "Docker Compose v2 plugin is required (docker compose, not docker-compose)."
  log_error "Install it from: https://docs.docker.com/compose/install/"
  exit 1
fi
log_success "Docker Compose v2 plugin detected"

# ── Python version guard ──────────────────────────────────────────────────────
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_MINOR=11
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt "$REQUIRED_MINOR" ]]; then
  log_error "Python 3.${REQUIRED_MINOR}+ is required (found $PYTHON_VERSION)."
  exit 1
fi

# ── Node version guard ────────────────────────────────────────────────────────
NODE_VERSION=$(node --version | sed 's/v//')
NODE_MAJOR=$(echo "$NODE_VERSION" | cut -d. -f1)
if [[ "$NODE_MAJOR" -lt 20 ]]; then
  log_warn "Node.js 20+ is recommended (found v${NODE_VERSION}). Proceeding anyway."
fi

# ── Repo root check ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$REPO_ROOT/docker-compose.yml" ]]; then
  log_error "Could not locate docker-compose.yml at repo root ($REPO_ROOT)."
  log_error "Ensure you are running this script from within the cloned AlgoDollar repository."
  exit 1
fi

cd "$REPO_ROOT"
log_info "Repo root: $REPO_ROOT"

# ── Backend Python install ────────────────────────────────────────────────────
if [[ -d "$REPO_ROOT/backend" ]]; then
  log_info "Installing backend Python dependencies..."
  python3 -m pip install --upgrade pip --quiet
  python3 -m pip install -e "$REPO_ROOT/backend[test]" --quiet
  log_success "Backend Python dependencies installed."
else
  log_warn "backend/ directory not found — skipping Python install."
fi

# ── Frontend Node install ─────────────────────────────────────────────────────
if [[ -d "$REPO_ROOT/frontend" && -f "$REPO_ROOT/frontend/package.json" ]]; then
  log_info "Installing frontend Node.js dependencies..."
  npm ci --prefix "$REPO_ROOT/frontend" --quiet
  log_success "Frontend Node.js dependencies installed."
else
  log_warn "frontend/package.json not found — skipping npm install."
fi

# ── .env file ─────────────────────────────────────────────────────────────────
if [[ ! -f "$REPO_ROOT/.env" ]]; then
  cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
  log_success "Created .env from .env.example"
  log_warn "Edit .env and fill in your real credentials before starting services."
else
  log_info ".env already exists — skipping copy."
fi

# ── Next steps ────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Setup complete! Next steps:${RESET}"
echo ""
echo -e "  1. ${CYAN}Edit .env${RESET} and set your KITE_API_KEY, KITE_API_SECRET, and SECRET_KEY."
echo -e "  2. ${CYAN}make dev-up${RESET}        — start all Docker services"
echo -e "  3. ${CYAN}make migrate${RESET}       — run database migrations"
echo -e "  4. ${CYAN}make seed-paper${RESET}    — populate paper-trading demo data (optional)"
echo -e "  5. Open ${CYAN}http://localhost:3000${RESET} in your browser"
echo ""
echo -e "Run ${CYAN}make help${RESET} to see all available commands."
echo ""
