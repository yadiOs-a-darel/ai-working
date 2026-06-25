.PHONY: install install-conda start start-voice start-text start-api start-ui test lint clean

RUN = bash scripts/run.sh

install:
	bash scripts/install.sh

# Installation minimale pour environnement Conda (sans apt upgrade ni systemd)
install-conda:
	SKIP_APT_UPGRADE=1 SKIP_SYSTEMD=1 REQUIRE_CONDA=1 bash scripts/install.sh

start:
	$(RUN) python main.py

start-voice:
	$(RUN) python main.py --mode voice

start-text:
	$(RUN) python main.py --mode text

start-api:
	$(RUN) uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

start-ui:
	$(RUN) python main.py --mode ui

test:
	$(RUN) pytest tests/ -v

lint:
	$(RUN) ruff check . && mypy .

clean:
	find . -type d -name __pycache__ | xargs rm -rf 2>/dev/null || true
	find . -name "*.pyc" | xargs rm -f 2>/dev/null || true
