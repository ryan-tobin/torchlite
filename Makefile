.PHONY: help install install-dev test test-cov lint format type-check docs clean build upload

help:
	@echo "Available commands:"
	@echo "  install       Install the package"
	@echo "  install-dev   Install with development dependencies"
	@echo "  test          Run tests"
	@echo "  test-cov      Run tests with coverage"
	@echo "  lint          Run linters"
	@echo "  format        Format code"
	@echo "  type-check    Run type checking"
	@echo "  docs          Build documentation"
	@echo "  clean         Clean build artifacts"
	@echo "  build         Build distribution packages"
	@echo "  upload        Upload to PyPI"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=torchlite --cov-report=html --cov-report=term

lint:
	flake8 torchlite tests examples
	pylint torchlite || true
	bandit -r torchlite -f json -o bandit_report.json || true

format:
	black torchlite tests examples
	isort torchlite tests examples

type-check:
	mypy torchlite --ignore-missing-imports

docs:
	cd docs && make clean && make html

clean:
	rm -rf build dist *.egg-info
	rm -rf .coverage htmlcov .pytest_cache
	rm -rf docs/_build
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

# Development workflow
dev: install-dev
	@echo "Development environment ready!"

# Run all checks
check: format lint type-check test
	@echo "All checks passed!"
