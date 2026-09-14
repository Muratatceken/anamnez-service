.PHONY: test run dev-ollama bundle docker-build lint
PY ?= .venv/bin/python

test:            ## Birim + regresyon testleri (Ollama gerekmez)
	$(PY) -m pytest tests/ -q

run:             ## Servisi lokal başlat (config/service.yaml)
	ANAMNEZ_SERVER__API_KEY=$${API_KEY:-devkey} $(PY) -m service

dev-ollama:      ## Mac geliştirme: ARM64 Ollama + modeller
	/opt/homebrew/bin/ollama serve & sleep 3; ollama pull glm-ocr; ollama pull qwen3:8b

bundle:          ## Offline kurulum paketi (internetli makinede)
	scripts/prepare_models.sh bundle/

docker-build:
	docker build -f deploy/Dockerfile -t anamnez-service:1.1.0 .

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*##' '{printf "  %-14s %s\n", $$1, $$2}'
