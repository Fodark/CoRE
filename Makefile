SHELL := /bin/bash

help: ## Show this help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-40s\033[0m %s\n", $$1, $$2}'

clean:  ## Remove all generated files
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . -type f -name "eval.log" -ls -delete
	find . -type f -name "train.log" -ls -delete
	find . | grep -E ".pytype" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E "__MACOSX" | xargs rm -rf
	find data -type f -name "*.tar" -ls -delete
	find data -type f -name "*.tar.gz" -ls -delete
	find data -type f -name "*.tgz" -ls -delete
	find data -type f -name "*.zip" -ls -delete
	rm -f .coverage
	rm -rf models/**

clean-logs:  ## Remove all logs
	rm -rf logs/**

format:  ## Format code
	pre-commit run --all-files

install:  ## Install dependencies
	if [! -d ".venv"]; then python3 -m venv .venv; fi

	source .venv/bin/activate; \
	pip install --upgrade pip; \
	pip install -r requirements.txt; \
	deactivate

install-dev: install  ## Install development dependencies
	source .venv/bin/activate; \
	pip install --upgrade pip; \
	pip install -r requirements-dev.txt; \
	deactivate

git-lfs:
	git lfs install

test:  ## Run tests
	source .venv/bin/activate; \
	pytest; \
	deactivate
