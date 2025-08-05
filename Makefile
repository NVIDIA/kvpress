SHELL := /bin/bash
UV ?= $(shell which uv)
BUILD_VERSION:=$(APP_VERSION)
TESTS_FILTER:=

PYTEST_LOG=--log-cli-level=debug --log-format="%(asctime)s %(levelname)s [%(name)s:%(filename)s:%(lineno)d] %(message)s" --log-date-format="%Y-%m-%d %H:%M:%S"

.PHONY: isort
isort:
	$(UV) run isort .

.PHONY: black
black:
	$(UV) run black .

PHONY: format
format: isort black

.PHONY: style
style: reports
	@echo -n > reports/flake8_errors.log
	@echo -n > reports/mypy_errors.log
	@echo -n > reports/mypy.log
	@echo -n > reports/copyright_errors.log
	@echo

	-$(UV) run flake8 | tee -a reports/flake8_errors.log
	@if [ -s reports/flake8_errors.log ]; then exit 1; fi

	-$(UV) run mypy . --check-untyped-defs | tee -a reports/mypy.log
	@if ! grep -Eq "Success: no issues found in [0-9]+ source files" reports/mypy.log ; then exit 1; fi

	@echo "Checking for SPDX-FileCopyrightText headers in Python files..."
	@find . -name "*.py" -not -path "*/\.*" | xargs grep -L "SPDX-FileCopyrightText:" | tee reports/copyright_errors.log || true
	@if [ -s reports/copyright_errors.log ]; then echo "Error: Missing SPDX-FileCopyrightText headers in files listed above"; exit 1; fi
	@echo "Success: All Python files have SPDX-FileCopyrightText headers."


reports:
	mkdir -p reports

.PHONY: test
test: reports
	PYTHONPATH=. \
	$(UV) run pytest \
		--cov-report xml:reports/coverage.xml \
		--cov=kvpress/ \
		--junitxml=./reports/junit.xml \
		tests/
