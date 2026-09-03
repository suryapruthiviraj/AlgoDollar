#!/usr/bin/env bash
# scripts/migrate.sh — Run Alembic database migrations
set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
log_success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
log_error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }

# ── Header ────────────────────────────────────────────────────────────────────
echo -e "\nAlgoDollar — Database Migration\n"

# ── Load .env if present ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [[ -f "$ENV_FILE" ]]; then
  log_info "Loading environment from $ENV_FILE"
  # Export only non-comment, non-empty lines
  set -o allexport
  # shellcheck disable=SC1090
  source <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$ENV_FILE")
  set +o allexport
fi

# ── DATABASE_URL guard ────────────────────────────────────────────────────────
if [[ -z "${DATABASE_URL:-}" ]]; then
  log_error "DATABASE_URL is not set."
  log_error "Set it in your .env file or export it before running this script."
  echo ""
  echo "Example:"
  echo "  export DATABASE_URL=postgresql+asyncpg://algodollar:algodollar_password@localhost:5432/algodollar"
  exit 1
fi

log_info "DATABASE_URL: ${DATABASE_URL%%@*}@<redacted>"

# ── Alembic check ─────────────────────────────────────────────────────────────
if ! command -v alembic &>/dev/null; then
  log_error "alembic not found in PATH."
  log_error "Install it with: pip install -e \"backend[test]\" or activate your virtual environment."
  exit 1
fi

# ── Change to backend directory ───────────────────────────────────────────────
BACKEND_DIR="$REPO_ROOT/backend"

if [[ ! -d "$BACKEND_DIR" ]]; then
  log_error "backend/ directory not found at $BACKEND_DIR"
  exit 1
fi

cd "$BACKEND_DIR"

# ── Confirm current revision ──────────────────────────────────────────────────
log_info "Current migration state:"
alembic current 2>&1 | sed 's/^/  /'

# ── Run migrations ────────────────────────────────────────────────────────────
log_info "Running: alembic upgrade head"
alembic upgrade head

# ── Confirm new state ─────────────────────────────────────────────────────────
log_info "Migration state after upgrade:"
alembic current 2>&1 | sed 's/^/  /'

log_success "All migrations applied successfully."
echo ""
