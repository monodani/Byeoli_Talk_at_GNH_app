PY=python
PIP=pip
RUFF=ruff

.PHONY: setup lock lint run build-index test validate-schema test-performance clean clean-all

setup:
	$(PIP) install -r requirements.txt
	@mkdir -p cache logs vectorstores reports
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

# --- Schema validation (lightweight, no external deps required) ---
# Validates that JSON files in schemas/ are well-formed and look like JSON Schema.
# If 'jsonschema' package is installed, also validates against the Draft2020-12 meta-schema.
validate-schema:
	@echo "[validate-schema] validating JSON schemas in ./schemas"
	@$(PY) - <<'PY'
import json, os, sys, importlib.util
from pathlib import Path

schemas_dir = Path("schemas")
if not schemas_dir.exists():
    print("[validate-schema] SKIP: ./schemas not found"); sys.exit(0)

def has_jsonschema():
    return importlib.util.find_spec("jsonschema") is not None

schemas = list(schemas_dir.glob("*.json"))
if not schemas:
    print("[validate-schema] WARN: no *.json files under ./schemas")
    sys.exit(0)

err = 0
for p in schemas:
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        # basic shape checks
        if not isinstance(data, dict):
            raise ValueError("schema root must be an object")
        # soft checks
        hints = []
        if "$schema" not in data:
            hints.append("missing '$schema'")
        if "type" not in data:
            hints.append("missing 'type'")
        if data.get("type") == "object" and "properties" not in data:
            hints.append("missing 'properties' for object")
        if hints:
            print(f"[validate-schema] HINT {p}: " + "; ".join(hints))
        else:
            print(f"[validate-schema] OK   {p}")
    except Exception as e:
        err += 1
        print(f"[validate-schema] FAIL {p}: {e}")

# optional strict validation with jsonschema if available
if has_jsonschema():
    try:
        import jsonschema
        from jsonschema.validators import validator_for
        metas = [
            "https://json-schema.org/draft/2020-12/schema",
            "https://json-schema.org/draft/2019-09/schema",
            "http://json-schema.org/draft-07/schema#",
        ]
        for p in schemas:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("$schema")
            if meta not in metas:
                # still try to pick a sensible default
                meta = metas[0]
            # fetch meta (jsonschema bundles common metas)
            try:
                cls = validator_for({"$schema": meta, "type": "object"})
                cls.check_schema(data)  # raises on invalid schema
                print(f"[validate-schema] STRICT OK {p}")
            except Exception as se:
                err += 1
                print(f"[validate-schema] STRICT FAIL {p}: {se}")
    except Exception as e:
        print(f"[validate-schema] STRICT SKIP: jsonschema error {e}")

sys.exit(1 if err else 0)
PY

# --- Lightweight performance smoke test (p50/p95 placeholders) ---
# Produces reports/perf_YYYYMMDD.json with timing skeleton so CI can archive.
# Real handler-level metrics should be emitted by the app; this is a safe stub.
test-performance:
	@echo "[perf] running lightweight performance smoke test"
	@$(PY) - <<'PY'
import json, os, time, statistics as st
from datetime import datetime
os.makedirs("reports", exist_ok=True)

# Stub timings (replace later by real hook to your router/handlers)
samples = []
for i in range(10):
    t0 = time.perf_counter()
    # simulate timebox budget split: routing 0.4s + handlers 1.1s (but we don't actually sleep that long)
    time.sleep(0.01)  # simulate tiny work
    dt = (time.perf_counter() - t0) * 1000
    samples.append(dt)

report = {
    "ts": datetime.utcnow().isoformat()+"Z",
    "n": len(samples),
    "p50_ms": round(st.median(samples), 2),
    "p95_ms": round(sorted(samples)[int(0.95*len(samples))-1], 2),
    "first_token_ms": None,          # fill from app logs if available
    "low_confidence_rate": None,     # fill from app logs if available
    "notes": "stub harness; wire to app metrics for real values"
}
fn = f"reports/perf_{datetime.utcnow().strftime('%Y%m%d')}.json"
with open(fn, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"[perf] wrote {fn}")
PY

# --- Unit tests ---
test:
	pytest -q --maxfail=1 --disable-warnings

# --- Cleanup ---
clean:
	rm -rf cache/* logs/* reports/*

# Danger: also clears vectorstores (use with care)
clean-all:
	rm -rf cache/* logs/* reports/* vectorstores/*
