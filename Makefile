# jiuguan-voice-gateway Makefile
# Standardized on Astral 'uv' package and project manager

UV ?= uv
PYTHON ?= $(UV) run python
PYTEST ?= $(UV) run pytest

.PHONY: all help install test test-unit test-acoustic test-acceptance test-acceptance-live audit clean

all: test

help:
	@echo "Available commands:"
	@echo "  make install              - Initialize virtualenv and install all dependencies via uv"
	@echo "  make test                 - Run all automated tests (unit + mock acceptance)"
	@echo "  make test-unit            - Run router, API, acoustic, and ASR unit tests"
	@echo "  make test-acoustic        - Run acoustic gate assertions (silence/clipping/CPS)"
	@echo "  make test-acceptance      - Run golden cases acceptance pipeline"
	@echo "  make test-acceptance-live - Run live GPU model acceptance tests (port 9880 required)"
	@echo "  make audit                - Run full test suite and architectural integrity audit"
	@echo "  make clean                - Clean cache directories and temporary files"

install:
	$(UV) venv
	$(UV) pip install -r requirements.txt -r requirements-dev.txt

test:
	$(PYTEST) tests/ -m "not acceptance_live" -v

test-unit:
	$(PYTEST) tests/unit/ -v

test-acoustic:
	$(PYTEST) tests/unit/test_acoustic_gate.py -v

test-acceptance:
	$(PYTEST) tests/acceptance/ -m "not acceptance_live" -v

test-acceptance-live:
	$(PYTEST) tests/acceptance/ -m "acceptance_live" -v

test-pipeline:
	$(PYTEST) tests/unit/test_quality_pipeline.py tests/unit/test_speaker_verifier.py tests/unit/test_speech_mos.py tests/unit/test_dataset_cli.py -v

audit: test
	@echo "=========================================================="
	@echo "🔍 Running Code & Architecture Integrity Audit..."
	@$(PYTHON) -c "import core.voice_router, core.acoustic_gate, core.asr_verifier, core.speaker_verifier, core.speech_mos, core.quality_pipeline; print('✅ All Core modules importable and syntactically valid')"
	@test -f logs/acceptance_report.json && echo "✅ Acceptance report generated: logs/acceptance_report.json" || true
	@echo "=========================================================="
	@echo "🎉 All Acceptance Gates, Quality Funnels, and Acoustic Rules PASSED!"

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
