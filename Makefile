.PHONY: install test lint train train-fast serve ui drift retrain docker mlflow-ui
install:       ; pip install -r requirements-dev.txt && pip install -e . --no-deps
lint:          ; ruff check src tests
test:          ; pytest -q --cov=spamguard --cov-report=term-missing
train:         ; python -m spamguard.train
train-fast:    ; python -m spamguard.train --fast
serve:         ; uvicorn spamguard.api:app --reload --port 8000
ui:            ; python -m spamguard.app_gradio
drift:         ; python -m spamguard.monitoring
retrain:       ; python -m spamguard.retrain --promote
docker:        ; docker compose up --build
mlflow-ui:     ; mlflow ui --backend-store-uri sqlite:///mlflow.db
