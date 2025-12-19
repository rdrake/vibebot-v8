.PHONY: install run test lint format format-check typecheck check ci clean deep-clean setup-http help \
       docker-build docker-run install-service uninstall-service install-timer uninstall-timer install-hooks pre-commit

install:
	uv sync

install-hooks:
	uv run pre-commit install
	@echo "Git hooks installed"

pre-commit:
	uv run pre-commit run --all-files

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

check: lint format-check typecheck test

ci:
	uv sync --locked
	uv run pre-commit run --all-files
	$(MAKE) test

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
	@echo "  check           - Run all checks (lint, format-check, typecheck, test)"
	@echo "  ci              - Run CI checks (sync --locked, pre-commit, test with coverage)"
	@echo "  clean           - Remove cache files"
	@echo "  deep-clean      - Remove venv and uv cache (full reset)"
	@echo "  setup-http      - Create HTTP directory for code/image output"
	@echo "  docker-build    - Build Docker image locally"
	@echo "  docker-run      - Run Docker container locally"
	@echo "  install-service - Install systemd user service"
	@echo "  uninstall-service - Remove systemd user service"
	@echo "  install-timer   - Install auto-update timer (checks GHCR every 15 min)"
	@echo "  uninstall-timer - Remove auto-update timer"

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
	@echo "Copying example config..."
	cp .env.example ~/.config/vibebot/env
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
