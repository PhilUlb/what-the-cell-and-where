.PHONY: help setup install install-gpu install-dev install-docs clean lint format typecheck test test-cov train eval predict preprocess docs

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install pre-commit hooks and setup development environment
	poetry install --with dev
	pre-commit install
	@echo "Development environment setup complete!"

install: ## Install the package (CPU version)
	poetry install --without gpu

install-gpu: ## Install the package with GPU support
	poetry install --with gpu

install-dev: ## Install development dependencies
	poetry install --with dev

install-docs: ## Install documentation dependencies
	poetry install --with docs

clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

lint: ## Run linting checks
	poetry run ruff check .

format: ## Format code with black and isort
	poetry run black .
	poetry run ruff check --fix .

typecheck: ## Run type checking with mypy
	poetry run mypy wtcell/

test: ## Run tests
	poetry run pytest

test-cov: ## Run tests with coverage
	poetry run pytest --cov=wtcell --cov-report=html --cov-report=term

preprocess: ## Run preprocessing pipeline
	poetry run wtcell preprocess --config configs/data.yaml --source rle

train: ## Train a model
	poetry run wtcell train --config configs/data.yaml configs/model.yaml configs/train.yaml

eval: ## Evaluate a trained model
	poetry run wtcell evaluate --config configs/eval.yaml --ckpt runs/latest/best.ckpt

predict: ## Run inference on new images
	poetry run wtcell predict --config configs/predict.yaml --input data/test_images --output data/predictions

docs: ## Build documentation
	poetry run mkdocs build

docs-serve: ## Serve documentation locally
	poetry run mkdocs serve

docker-build: ## Build Docker image
	docker build -t wtcell:latest .

docker-run: ## Run Docker container
	docker run --gpus all -it wtcell:latest

docker-run-cpu: ## Run Docker container (CPU only)
	docker run -it wtcell:latest

ci: ## Run CI checks locally
	poetry run ruff check .
	poetry run black --check .
	poetry run mypy wtcell/
	poetry run pytest 