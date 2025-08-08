PY=python
PIP=pip
RUFF=ruff

.PHONY: setup lock lint run build-index test clean

setup:
	$(PIP) install -r requirements.txt
	@mkdir -p cache logs vectorstores
	@echo "[setup] done"

lock:
	$(PIP) freeze | sed '/@ file:\/\//d' > requirements.lock.txt

lint:
	$(RUFF) check . --fix

run:
	streamlit run app.py

# --- Index build entrypoints ---
build-index:
	$(PY) -m modules.loader_publish
	$(PY) -m modules.loader_satisfaction.loader_course_satisfaction
	$(PY) -m modules.loader_satisfaction.loader_subject_satisfaction
	$(PY) -m modules.loader_general
	$(PY) -m modules.loader_cyber
	$(PY) -m modules.loader_notice
	$(PY) -m modules.loader_menu

# --- Tests ---
test:
	pytest -q --maxfail=1 --disable-warnings

clean:
	rm -rf cache/* logs/*
