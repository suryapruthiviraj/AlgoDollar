# AlgoDollar Deployment Guide

> **CRITICAL**: Never auto-enable live trading on deploy. Live trading must
> always require explicit manual intervention. Read the live trading section
> before configuring any production environment.

---

## Table of Contents
1. [Docker Compose (Development)](#1-docker-compose-development)
2. [Production Architecture](#2-production-architecture)
3. [Static IP Requirement](#3-static-ip-requirement)
4. [Environment Security](#4-environment-security)
5. [Database Backup](#5-database-backup)
6. [Monitoring Setup](#6-monitoring-setup)
7. [CI/CD Pipeline](#7-cicd-pipeline)
8. [Live Trading Gates](#8-live-trading-gates)

---

## 1. Docker Compose (Development)

### Prerequisites

- Docker Engine 24+
- Docker Compose v2.24+
- 4 GB RAM minimum (8 GB recommended)
- 10 GB free disk space

### Start All Services

```bash
# Copy example env and configure
cp .env.example .env
# Edit .env — at minimum set SECRET_KEY

# Start all services in the background
make dev-up

# Check all services are healthy
docker compose ps

# Follow backend logs
make logs-backend

# Follow all logs
make logs
```

### Service Ports (Development)

| Service | Port | Purpose |
|---|---|---|
| PostgreSQL | 5432 | Database (internal) |
| Redis | 6379 | Cache + broker (internal) |
| Backend (FastAPI) | 8000 | REST API + WebSocket |
| Frontend (Next.js) | 3000 | Dashboard UI |
| Celery Flower | 5555 | Task monitoring (optional) |

### Common Make Targets

```bash
make dev-up          # Start all services
make dev-down        # Stop all services
make dev-restart     # Restart backend only (after code changes)
make migrate         # Run alembic database migrations
make seed            # Load seed/fixture data
make test            # Run backend tests
make test-cov        # Run tests with coverage report
make lint            # Run ruff + mypy
make logs            # Tail all service logs
make logs-backend    # Tail backend logs only
make shell-db        # Open psql shell to the development database
make shell-redis     # Open redis-cli
```

### Resetting Development State

```bash
# Stop everything and delete volumes (WARNING: destroys all data)
docker compose down -v

# Then start fresh
make dev-up && make migrate
```

---

## 2. Production Architecture

### Minimum Production Stack

```
┌─────────────────────────────────────────────────────────────┐
│                   Reverse Proxy (nginx / Caddy)            │
│           TLS termination, static file serving             │
│                      STATIC IP REQUIRED                    │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────▼────────────────┐
          │   FastAPI (uvicorn)           │
          │   2+ workers (--workers 2)    │
          │   gunicorn process manager    │
          └──────────────┬────────────────┘
                         │
       ┌─────────────────┼──────────────────┐
       ▼                 ▼                  ▼
 PostgreSQL 16       Redis 7          Celery Workers
 (separate VM        (separate VM     (2–4 workers)
  or RDS)             or managed)
```

### Recommended VM Specs (Single Server, Personal Use)

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 2 vCPU | 4 vCPU |
| RAM | 4 GB | 8 GB |
| Storage | 50 GB SSD | 100 GB SSD |
| Network | Static IP | Static IP |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |

### Systemd Service Files

**FastAPI backend** (`/etc/systemd/system/algodollar-api.service`):

```ini
[Unit]
Description=AlgoDollar FastAPI Backend
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=algodollar
WorkingDirectory=/opt/algodollar/backend
EnvironmentFile=/opt/algodollar/.env
ExecStart=/opt/algodollar/venv/bin/gunicorn app.main:app \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 127.0.0.1:8000 \
    --access-logfile /var/log/algodollar/access.log \
    --error-logfile /var/log/algodollar/error.log
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Celery worker** (`/etc/systemd/system/algodollar-worker.service`):

```ini
[Unit]
Description=AlgoDollar Celery Worker
After=network.target redis.service

[Service]
Type=exec
User=algodollar
WorkingDirectory=/opt/algodollar/backend
EnvironmentFile=/opt/algodollar/.env
ExecStart=/opt/algodollar/venv/bin/celery \
    -A app.celery_app worker \
    --loglevel=info \
    --concurrency=2
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Celery Beat scheduler** (`/etc/systemd/system/algodollar-beat.service`):

```ini
[Unit]
Description=AlgoDollar Celery Beat Scheduler
After=network.target redis.service

[Service]
Type=exec
User=algodollar
WorkingDirectory=/opt/algodollar/backend
EnvironmentFile=/opt/algodollar/.env
ExecStart=/opt/algodollar/venv/bin/celery \
    -A app.celery_app beat \
    --loglevel=info \
    --scheduler django_celery_beat.schedulers.DatabaseScheduler
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## 3. Static IP Requirement

Zerodha Kite Connect requires a static IP for production API usage. Without a
static IP, API requests may be rejected after IP changes.

### AWS Setup

```bash
# 1. Launch EC2 instance (t3.medium or larger recommended)
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \  # Ubuntu 22.04 LTS
  --instance-type t3.medium \
  --key-name your-key-pair

# 2. Allocate Elastic IP
aws ec2 allocate-address --domain vpc

# 3. Associate EIP with instance
aws ec2 associate-address \
  --instance-id i-xxxxxxxxx \
  --allocation-id eipalloc-xxxxxxxxx

# 4. Note the Elastic IP — add this to Kite Connect app settings
```

### Hetzner Cloud Setup (cost-effective alternative)

```bash
# 1. Create a server (CX21 or CX31 recommended)
hcloud server create \
  --name algodollar-prod \
  --type cx21 \
  --image ubuntu-22.04 \
  --location nbg1

# 2. Get the server IP (it is static by default on Hetzner)
hcloud server describe algodollar-prod | grep "Public Net"

# 3. Add this IP to Kite Connect app settings at developers.kite.trade
```

---

## 4. Environment Security

### Required Security Practices

1. **Never commit secrets**:
   - `.env` is in `.gitignore` — verify before every commit
   - Use `git status` and `git diff --cached` before `git commit`

2. **Minimum SECRET_KEY length**:
   ```bash
   # Generate a strong secret key
   openssl rand -hex 32
   ```

3. **Database access**:
   - PostgreSQL must NOT be publicly accessible
   - Use VPC security groups or firewall rules to restrict to application server IP only
   - Enable SSL: `DATABASE_URL=postgresql+asyncpg://user:pass@host/db?ssl=require`

4. **Redis access**:
   - Redis must NOT be publicly accessible
   - Enable Redis AUTH: `REDIS_URL=redis://:your_redis_password@host:6379/0`

5. **API keys rotation**:
   - Rotate `ZERODHA_API_SECRET` immediately if ever exposed
   - Rotate `SECRET_KEY` requires re-login for all active users
   - Document when each key was last rotated

6. **File permissions**:
   ```bash
   chmod 600 /opt/algodollar/.env
   chown algodollar:algodollar /opt/algodollar/.env
   ```

---

## 5. Database Backup

### Automated PostgreSQL Backup

```bash
# /opt/algodollar/scripts/backup_db.sh
#!/bin/bash
set -euo pipefail

BACKUP_DIR="/opt/algodollar/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_NAME="algodollar"
BACKUP_FILE="${BACKUP_DIR}/algodollar_${TIMESTAMP}.sql.gz"

mkdir -p "$BACKUP_DIR"

# Dump and compress
PGPASSWORD="$DB_PASSWORD" pg_dump \
  -h localhost \
  -U algodollar \
  -d "$DB_NAME" \
  --no-owner \
  --no-acl \
  | gzip > "$BACKUP_FILE"

# Keep only last 30 days of backups
find "$BACKUP_DIR" -name "algodollar_*.sql.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_FILE"
```

Schedule via cron (daily at 02:00 IST):

```bash
# crontab -e
30 20 * * * /opt/algodollar/scripts/backup_db.sh >> /var/log/algodollar/backup.log 2>&1
# Note: 20:30 UTC = 02:00 IST
```

### Restore from Backup

```bash
# Stop all services first
systemctl stop algodollar-api algodollar-worker algodollar-beat

# Restore
gunzip -c /opt/algodollar/backups/algodollar_20240115_020000.sql.gz \
  | PGPASSWORD="$DB_PASSWORD" psql -h localhost -U algodollar algodollar

# Restart services
systemctl start algodollar-api algodollar-worker algodollar-beat
```

---

## 6. Monitoring Setup

### Health Checks

```bash
# API health check
curl -s http://localhost:8000/health
# Expected: {"status": "healthy", "db": "ok", "redis": "ok"}

# Set up monitoring ping (e.g., Uptime Robot or Better Uptime)
# Ping: https://yourdomain.com/health every 1 minute
```

### Log Management

```bash
# Application logs (structured JSON via structlog)
tail -f /var/log/algodollar/error.log | jq '.'

# Key log events to monitor:
#   event: "order_placed"       — normal operation
#   event: "order_rejected"     — safety check triggered
#   event: "kill_switch_active" — trading halted
#   event: "max_drawdown_hit"   — risk limit triggered
#   event: "token_expired"      — needs manual login
#   level: "error"              — needs immediate attention
```

### Prometheus Metrics (Optional)

```python
# Add to FastAPI startup
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app, endpoint="/metrics")
```

Key metrics to track:
- `orders_placed_total` — order volume
- `orders_rejected_total{reason}` — safety rejections by type
- `portfolio_value` — portfolio gauge
- `strategy_pnl{strategy}` — per-strategy P&L
- `drawdown_pct` — current drawdown gauge

---

## 7. CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          cd backend
          pip install -e ".[test]"

      - name: Run linter
        run: |
          cd backend
          ruff check .
          mypy app/ --ignore-missing-imports

      - name: Run tests
        run: |
          cd backend
          pytest tests/ -v --cov=app --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t algodollar-backend ./backend

      - name: Verify image starts
        run: |
          docker run --rm -e DATABASE_URL=sqlite:///test.db \
            algodollar-backend python -c "from app.main import app; print('OK')"
```

### Deployment to Production

Deployments are intentionally manual:

```bash
# 1. Pull latest code
cd /opt/algodollar
git pull origin main

# 2. Install any new dependencies
source venv/bin/activate
pip install -e backend/

# 3. Review and apply migrations manually
alembic -c backend/alembic.ini upgrade head

# 4. Restart services
systemctl restart algodollar-api algodollar-worker algodollar-beat

# 5. Verify health
curl http://localhost:8000/health

# 6. Review logs for errors
journalctl -u algodollar-api -n 50 --no-pager
```

---

## 8. Live Trading Gates

> **NEVER enable live trading automatically during deployment.**

Live trading requires ALL of the following conditions to be met simultaneously:

### Required Configuration

```env
# Both of these must be explicitly set to enable live trading
TRADING_MODE=live
LIVE_TRADING_ENABLED=true

# Valid Zerodha credentials (not test values)
ZERODHA_API_KEY=<real_api_key>
ZERODHA_API_SECRET=<real_api_secret>
```

### Required Runtime State

1. **Valid access token**: A fresh Kite Connect access token must exist in Redis.
   The system will NOT start live trading without one.

2. **Kill switch is OFF**: `KILL_SWITCH=false` and the kill switch is not active
   in the database.

3. **Static IP confirmed**: API calls must originate from the whitelisted static IP.

4. **Risk limits set**: `MAX_PORTFOLIO_LOSS_PCT` and `MAX_SINGLE_POSITION_PCT` must
   be explicitly configured (defaults are intentionally conservative).

### Recommended Pre-Live Checklist

```
[ ] Paper trading has run for at least 30 trading days
[ ] Backtest results reviewed and understood (including OOS performance)
[ ] Walk-forward validation shows consistent positive results
[ ] Cost model validated against actual Zerodha brokerage statements
[ ] Risk limits set to conservative values (e.g., 10% max drawdown to start)
[ ] Kill switch tested: verify it blocks orders when activated
[ ] Daily token refresh tested: automated login working reliably
[ ] Static IP confirmed: Kite Connect API calls succeed from production IP
[ ] Alert webhook tested: kill switch alert reaches notification channel
[ ] Backup and restore tested: database can be restored from backup
[ ] Emergency contact: know how to manually square off all positions via Zerodha Console
```

### Emergency Procedures

**To immediately halt all trading**:

```bash
# Option 1: API call
curl -X POST http://localhost:8000/api/v1/risk/kill-switch/activate \
  -H "Authorization: Bearer your_admin_token"

# Option 2: Environment variable (requires service restart)
echo "KILL_SWITCH=true" >> /opt/algodollar/.env
systemctl restart algodollar-api algodollar-worker algodollar-beat

# Option 3: If all else fails — manual square off via Zerodha Console
# Log in to zerodha.com -> Console -> Positions -> Exit all positions
```

**To re-enable trading after a halt**:
- Identify and fix the cause of the halt
- Review all positions and P&L
- Update risk limits if necessary
- Set `KILL_SWITCH=false` via API or environment variable
- Verify the system is healthy before re-enabling
