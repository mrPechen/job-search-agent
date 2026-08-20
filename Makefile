run:
	uv run uvicorn main:app --reload

test:
	uv run pytest

lint:
	uv run black --check src tests config.py main.py

format:
	uv run black src tests config.py main.py
