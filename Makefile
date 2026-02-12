.PHONY: install run test lint format format-check typecheck syntax-check check preflight ci clean deep-clean setup-http help \
       docker-build docker-run install-service uninstall-service install-timer uninstall-timer install-hooks pre-commit \
       install-deploy worktree-create worktree-remove wait-ci rebase-pr

install:
	uv sync

install-hooks:
	uv run prek install
	@echo "Git hooks installed"

pre-commit:
	uv run prek run --all-files

run:
	uv run limnoria bot.conf

test:
	uv run pytest plugins/llm/tests/ -v --cov --cov-report=term-missing --cov-fail-under=80

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run ty check plugins/llm/src/

syntax-check:
	uv run python scripts/check_python_syntax_compat.py --versions 3.12 3.13 3.14

check: lint format-check typecheck syntax-check test

preflight: format check

ci:
	uv sync --locked
	uv run prek run --all-files
	$(MAKE) syntax-check
	$(MAKE) test

# Worktree workflow
WORKTREE_DIR ?= .worktrees

worktree-create:
ifndef BRANCH
	$(error BRANCH is required: make worktree-create BRANCH=fix/my-fix)
endif
	@mkdir -p $(WORKTREE_DIR)
	git worktree add $(WORKTREE_DIR)/$(BRANCH) -b $(BRANCH)
	cd $(WORKTREE_DIR)/$(BRANCH) && uv sync
	@echo "Worktree ready at $(WORKTREE_DIR)/$(BRANCH)"

worktree-remove:
ifndef BRANCH
	$(error BRANCH is required: make worktree-remove BRANCH=fix/my-fix)
endif
	git worktree remove $(WORKTREE_DIR)/$(BRANCH)
	-git branch -d $(BRANCH)
	@echo "Worktree and branch $(BRANCH) removed"

# GitHub helpers
wait-ci:
	@RUN_ID=$$(gh run list --branch main --limit 1 --json databaseId --jq '.[0].databaseId'); \
	echo "Watching run $$RUN_ID …"; \
	while true; do \
		STATUS=$$(gh run view "$$RUN_ID" --json status,conclusion --jq '.status + " " + .conclusion'); \
		echo "$$(date +%H:%M:%S) $$STATUS"; \
		case "$$STATUS" in \
			completed\ success) echo "CI passed ✓"; exit 0;; \
			completed\ *) echo "CI failed ✗"; exit 1;; \
		esac; \
		sleep 10; \
	done

rebase-pr:
ifndef PR
	$(error PR is required: make rebase-pr PR=42)
endif
	gh pr comment $(PR) --body "@dependabot rebase"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ty_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

deep-clean: clean
	rm -rf .venv
	uv cache clean

setup-http:
	@echo "Creating HTTP directory for code/image output..."
	mkdir -p /var/www/llm
	chmod 755 /var/www/llm
	@echo "HTTP directory created at /var/www/llm"
	@echo "Configure your web server to serve this directory"
	@echo ""
	@echo "Example nginx config:"
	@echo "  location /llm {"
	@echo "    root /var/www;"
	@echo "    autoindex off;"
	@echo "  }"

help:
	@echo "Available targets:"
	@echo "  install         - Install dependencies with uv"
	@echo "  install-hooks   - Install git pre-commit hooks"
	@echo "  pre-commit      - Run pre-commit on all files"
	@echo "  run             - Start the bot"
	@echo "  test            - Run all tests"
	@echo "  lint            - Run ruff linter"
	@echo "  format          - Format code with ruff"
	@echo "  format-check    - Check formatting without changes"
	@echo "  typecheck       - Run ty type checker"
	@echo "  syntax-check    - Validate Python grammar compatibility (3.12-3.14)"
	@echo "  check           - Run all checks (lint, format-check, typecheck, syntax-check, test)"
	@echo "  preflight       - Auto-format then run all checks"
	@echo "  ci              - Run CI checks (sync --locked, pre-commit, test with coverage)"
	@echo "  worktree-create - Create isolated worktree (BRANCH=name required)"
	@echo "  worktree-remove - Remove worktree and branch (BRANCH=name required)"
	@echo "  wait-ci         - Watch current GitHub Actions run until completion"
	@echo "  rebase-pr       - Ask dependabot to rebase a PR (PR=number required)"
	@echo "  clean           - Remove cache files"
	@echo "  deep-clean      - Remove venv and uv cache (full reset)"
	@echo "  setup-http      - Create HTTP directory for code/image output"
	@echo "  docker-build    - Build Docker image locally"
	@echo "  docker-run      - Run Docker container locally"
	@echo "  install-service - Install systemd user service"
	@echo "  uninstall-service - Remove systemd user service"
	@echo "  install-timer   - Install auto-update timer (checks GHCR every 15 min)"
	@echo "  uninstall-timer - Remove auto-update timer"
	@echo "  install-deploy  - Install service and timer together"

# Docker
IMAGE_NAME ?= ghcr.io/rdrake/vibebot-v8
IMAGE_TAG ?= latest

docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-run:
	docker run --rm -it \
		--user $$(id -u):$$(id -g) \
		-v $(PWD)/bot.conf:/app/bot.conf:ro \
		-v $(PWD)/conf:/app/conf \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/logs:/app/logs \
		$(IMAGE_NAME):$(IMAGE_TAG)

# systemd user service installation
install-service:
	@echo "Creating directories..."
	mkdir -p ~/.config/systemd/user
	mkdir -p ~/.config/vibebot
	mkdir -p ~/.local/share/vibebot/{conf,data,logs}
	@echo "Installing systemd unit..."
	cp vibebot.service ~/.config/systemd/user/
	@if [ ! -f ~/.config/vibebot/env ]; then \
		echo "Copying example config..."; \
		cp .env.example ~/.config/vibebot/env; \
	else \
		echo "Keeping existing env file..."; \
	fi
	@if [ ! -f ~/.config/vibebot/bot.conf ]; then \
		echo "NOTE: Copy your bot.conf to ~/.config/vibebot/bot.conf"; \
	fi
	@echo "Reloading systemd..."
	systemctl --user daemon-reload
	@echo ""
	@echo "Installation complete. Next steps:"
	@echo "  1. Copy bot.conf to ~/.config/vibebot/bot.conf"
	@echo "  2. Edit ~/.config/vibebot/env with your API keys"
	@echo "  3. systemctl --user enable vibebot"
	@echo "  4. systemctl --user start vibebot"
	@echo "  5. loginctl enable-linger $$USER  (keeps service running after logout)"

uninstall-service:
	-systemctl --user stop vibebot
	-systemctl --user disable vibebot
	rm -f ~/.config/systemd/user/vibebot.service
	systemctl --user daemon-reload
	@echo "Service removed. Config files in ~/.config/vibebot/ preserved."

install-timer:
	@echo "Installing update timer..."
	mkdir -p ~/.config/systemd/user
	cp vibebot-updater.service vibebot-updater.timer ~/.config/systemd/user/
	systemctl --user daemon-reload
	systemctl --user enable --now vibebot-updater.timer
	@echo "Timer installed. Check status with: systemctl --user status vibebot-updater.timer"

uninstall-timer:
	-systemctl --user disable --now vibebot-updater.timer
	rm -f ~/.config/systemd/user/vibebot-updater.service
	rm -f ~/.config/systemd/user/vibebot-updater.timer
	systemctl --user daemon-reload
	@echo "Timer removed."

install-deploy: install-service install-timer
	@echo ""
	@echo "Deployment complete. Bot will auto-update every 15 minutes."
