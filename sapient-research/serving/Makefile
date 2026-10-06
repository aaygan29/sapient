# Sapient-1 serving layer — common commands.
# Mock mode (default) needs no GPU, no weights, no credentials.

.PHONY: setup dev test issue-key deploy fmt

setup:            ## install the API + mock-engine deps (editable)
	pip install -e ".[dev]"

dev:              ## run the API + playground locally in MOCK mode
	SAPIENT_ENGINE=mock SAPIENT_DOCS=1 uvicorn sapient_serving.api.app:app --reload --port 8000

test:             ## run the security + flow tests (mock engine)
	pytest -q

issue-key:        ## mint a client API key (prints the raw key once; stores only its hash)
	python scripts/issue_key.py

deploy:           ## deploy the real (GPU) service to Modal — requires the `real` extra + secrets
	modal deploy sapient_serving/modal_app.py
