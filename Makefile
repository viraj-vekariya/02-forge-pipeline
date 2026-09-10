# Forge Pipeline
#
#   make setup     install dependencies
#   make all       train -> evaluate -> export -> test  (the full pipeline)
#   make serve     run the inference service
#   make loadtest  ramp until SLA breach, then profile the cause
#   make report    consolidate every measured number into outputs/results.json

PY        ?= python3
JAVA_HOME ?= $(shell /usr/libexec/java_home -v 21 2>/dev/null || echo /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home)
MVN        = JAVA_HOME=$(JAVA_HOME) mvn
BASE      ?= http://127.0.0.1:8200

.PHONY: setup all train evaluate export test serve serve-features loadtest profile \
        report docker clean

setup:
	$(PY) -m pip install -r requirements.txt

all: train evaluate export test

train:
	$(PY) train/train.py

evaluate:
	$(PY) train/evaluate.py

export:
	$(PY) train/export.py

test:
	$(PY) -m pytest tests/ -q

serve:
	$(PY) -m uvicorn serve.app:app --host 127.0.0.1 --port 8200 --workers 1

serve-features:
	cd services/java-service && $(MVN) -q spring-boot:run

loadtest:
	$(PY) loadtest/harness.py --base $(BASE)
	$(PY) loadtest/profile.py

profile:
	$(PY) loadtest/profile.py

report:
	$(PY) loadtest/report.py

docker:
	docker compose up --build

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache \
	       services/java-service/target artifacts/model_traced.pt
