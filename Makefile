SHELL := /bin/bash
POETRY ?= $(shell which poetry)
BUILD_VERSION:=$(APP_VERSION)
TESTS_FILTER:=

PYTEST_LOG=--log-cli-level=debug --log-format="%(asctime)s %(levelname)s [%(name)s:%(filename)s:%(lineno)d] %(message)s" --log-date-format="%Y-%m-%d %H:%M:%S"

.PHONY: isort
isort:
	$(POETRY) run isort .

.PHONY: black
black:
	$(POETRY) run black .

PHONY: format
format: isort black

.PHONY: style
style: reports
	@echo -n > reports/flake8_errors.log
	@echo -n > reports/mypy_errors.log
	@echo -n > reports/mypy.log
	@echo

	-$(POETRY) run flake8 | tee -a reports/flake8_errors.log
	@if [ -s reports/flake8_errors.log ]; then exit 1; fi

	-$(POETRY) run mypy . --check-untyped-defs | tee -a reports/mypy.log
	@if ! grep -Eq "Success: no issues found in [0-9]+ source files" reports/mypy.log ; then exit 1; fi


reports:
	mkdir -p reports

.PHONY: test
test: reports
	PYTHONPATH=. \
	$(POETRY) run pytest \
		--cov-report xml:reports/coverage.xml \
		--cov=kvpress/ \
		--junitxml=./reports/junit.xml \
		tests/
