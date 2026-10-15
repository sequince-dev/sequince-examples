.DEFAULT_GOAL := help

DOCS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(DOCS_DIR)/..)

PIXI ?= pixi
EXAMPLE ?= gaussian
ENV ?= default

.PHONY: help setup test lint format-check check hooks example examples refresh docs docs-only validate

help:
	@printf '%s\n' \
	  'make setup         Refresh Sequince main and install environments' \
	  'make refresh       Resolve current Sequince main without sampling' \
	  'make check         Run tests, lint and formatting checks' \
	  'make test          Run pipeline tests' \
	  'make lint          Run Ruff lint checks' \
	  'make format-check  Check Python formatting' \
	  'make hooks         Run all pre-commit hooks (may edit files)' \
	  'make example EXAMPLE=gaussian [ENV=default]' \
	  '                   Run one example in the selected environment' \
	  'make examples      Run all documentation examples in their environments' \
	  'make docs          Reuse valid outputs or run examples, then build docs' \
	  'make docs-only     Build docs using existing validated outputs' \
	  'make validate      Run checks, all examples, then build docs'

refresh:
	cd "$(REPO_ROOT)" && $(PIXI) update sequince

setup: refresh
	cd "$(REPO_ROOT)" && $(PIXI) install --locked -e default -e docs -e check

test:
	cd "$(REPO_ROOT)" && $(PIXI) run --locked -e check test

lint:
	cd "$(REPO_ROOT)" && $(PIXI) run --locked -e check lint

format-check:
	cd "$(REPO_ROOT)" && $(PIXI) run --locked -e check format-check

check: test lint format-check

hooks:
	cd "$(REPO_ROOT)" && $(PIXI) run --locked -e check pre-commit run --all-files

example:
	cd "$(REPO_ROOT)" && $(PIXI) run --locked -e $(ENV) example $(EXAMPLE)

examples:
	cd "$(REPO_ROOT)" && $(PIXI) run --locked examples

docs: refresh
	@if ! (cd "$(REPO_ROOT)" && $(PIXI) run --locked -e docs validate-outputs); then \
	  $(MAKE) -C "$(DOCS_DIR)" examples || exit $$?; \
	fi
	$(MAKE) -C "$(DOCS_DIR)" docs-only

docs-only:
	cd "$(REPO_ROOT)" && $(PIXI) run --locked -e docs docs

# Keep execution before Sphinx even when invoked with make -j.
validate: check
	$(MAKE) -C "$(DOCS_DIR)" examples
	$(MAKE) -C "$(DOCS_DIR)" docs-only
