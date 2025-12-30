# Makefile for heteroage-clock

# Default target
.PHONY: all
all: install

# Install dependencies
.PHONY: install
install:
	@echo "Installing dependencies..."
	@python -m pip install --upgrade pip
	@pip install -e .

# Install development extras (for linting, testing, etc.)
.PHONY: dev
dev:
	@echo "Installing development dependencies..."
	@pip install -e ".[dev]"

# Run tests using pytest
.PHONY: test
test:
	@echo "Running tests..."
	@pytest --maxfail=5 --disable-warnings -q

# Lint the code using flake8
.PHONY: lint
lint:
	@echo "Linting the code..."
	@flake8 src tests

# Format the code using black
.PHONY: format
format:
	@echo "Formatting the code..."
	@black src tests

# Check the code style with flake8 and black
.PHONY: check
check: lint format

# Clean up unnecessary files
.PHONY: clean
clean:
	@echo "Cleaning up..."
	@rm -rf .tox .nox .coverage .hypothesis .pytest_cache .mypy_cache
	@rm -rf **/__pycache__ **/*.pyc **/*.pyo
	@rm -rf dist build *.egg-info
	@rm -rf .venv

# Run pre-commit hooks
.PHONY: precommit
precommit:
	@echo "Running pre-commit hooks..."
	@pre-commit run --all-files
