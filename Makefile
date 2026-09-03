.DEFAULT_GOAL := help

# ── Colours ───────────────────────────────────────────────────────────────────
BOLD   := \033[1m
RESET  := \033[0m
GREEN  := \033[32m
YELLOW := \033[33m
CYAN   := \033[36m

# ── Docker Compose shorthand ──────────────────────────────────────────────────
DC := docker compose

##@ Help

.PHONY: help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\n$(BOLD)AlgoDollar — Makefile targets$(RESET)\n\n"} \
		/^##@/ { printf "$(CYAN)%s$(RESET)\n", substr($$0,5) } \
		/^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

##@ Development

.PHONY: dev-up
dev-up: ## Start all services in detached mode
	@echo "$(YELLOW)Starting AlgoDollar services...$(RESET)"
	$(DC) up -d
	@echo "$(GREEN)Services started. Backend: http://localhost:8000  Frontend: http://localhost:3000$(RESET)"

.PHONY: dev-down
dev-down: ## Stop and remove all containers (preserves volumes)
	@echo "$(YELLOW)Stopping AlgoDollar services...$(RESET)"
	$(DC) down
	@echo "$(GREEN)Services stopped.$(RESET)"

.PHONY: logs
logs: ## Tail logs for all services (Ctrl-C to exit)
	$(DC) logs -f

.PHONY: shell-backend
shell-backend: ## Open a shell in the running backend container
	$(DC) exec backend /bin/bash

.PHONY: shell-db
shell-db: ## Open a psql session in the running postgres container
	$(DC) exec postgres psql -U algodollar -d algodollar

##@ Database

.PHONY: migrate
migrate: ## Run Alembic migrations (alembic upgrade head)
	@echo "$(YELLOW)Running database migrations...$(RESET)"
	$(DC) exec backend alembic upgrade head
	@echo "$(GREEN)Migrations complete.$(RESET)"

.PHONY: seed-paper
seed-paper: ## Seed the database with paper-trading demo data
	@echo "$(YELLOW)Seeding paper-trading data...$(RESET)"
	$(DC) exec backend python -m app.scripts.seed_paper
	@echo "$(GREEN)Paper data seeded.$(RESET)"

##@ Testing

.PHONY: test
test: test-backend test-frontend ## Run all tests (backend + frontend)

.PHONY: test-backend
test-backend: ## Run Python tests with coverage
	@echo "$(YELLOW)Running backend tests...$(RESET)"
	$(DC) exec backend pytest --cov=app --cov-report=term-missing -q
	@echo "$(GREEN)Backend tests complete.$(RESET)"

.PHONY: test-frontend
test-frontend: ## Run Next.js tests
	@echo "$(YELLOW)Running frontend tests...$(RESET)"
	$(DC) exec frontend npm test -- --passWithNoTests
	@echo "$(GREEN)Frontend tests complete.$(RESET)"

##@ Quality

.PHONY: lint
lint: ## Run ruff (Python) and ESLint (TypeScript)
	@echo "$(YELLOW)Linting Python...$(RESET)"
	$(DC) exec backend ruff check app
	@echo "$(YELLOW)Linting TypeScript...$(RESET)"
	$(DC) exec frontend npm run lint

.PHONY: type-check
type-check: ## Run mypy (Python) and tsc (TypeScript)
	@echo "$(YELLOW)Type-checking Python...$(RESET)"
	$(DC) exec backend mypy app
	@echo "$(YELLOW)Type-checking TypeScript...$(RESET)"
	$(DC) exec frontend npx tsc --noEmit

##@ Build & Release

.PHONY: build
build: ## Build Docker images for all services
	@echo "$(YELLOW)Building Docker images...$(RESET)"
	$(DC) build
	@echo "$(GREEN)Build complete.$(RESET)"

##@ Housekeeping

.PHONY: clean
clean: ## Remove containers, volumes, and orphan networks (DESTRUCTIVE)
	@echo "$(YELLOW)Warning: this will delete all Docker volumes including the database.$(RESET)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || (echo "Aborted." && exit 1)
	$(DC) down -v --remove-orphans
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Clean complete.$(RESET)"
