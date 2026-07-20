.PHONY: test unit-test debug install metadata smoke

test:
	uv run cube test knows-cube

unit-test:
	uv run pytest tests/

debug:
	uv run python -m knows_cube.debug

install:
	uv sync --all-extras
	uv pip install -e .

# Regenerates task_metadata.json, then verifies it in a FRESH process — the
# generator deliberately cannot import knows_cube (the config class auto-loads
# the very file being generated), so validation has to happen afterwards.
metadata:
	uv run python scripts/generate_task_metadata.py --force
	uv run python -c "from knows_cube.benchmark import KnowsBenchmarkConfig as C; \
	from knows_cube.task import KnowsTaskMetadata as M; t = C().tasks(); \
	assert len(t) == 110, len(t); \
	assert all(isinstance(v, M) for v in t.values()), 'lost _type discriminator'; \
	print(f'verified {len(t)} tasks')"

# Requires a gold-instance document id, full Google + Gemini credentials, and
# real network. SKIPs (exit 2) without credentials.
#   make smoke GOLD_DOC_ID=<id>
smoke:
	@test -n "$(GOLD_DOC_ID)" || { echo "usage: make smoke GOLD_DOC_ID=<document id>"; exit 1; }
	uv run python ../../scripts/smoke/knows_gold_eval.py --gold-doc-id "$(GOLD_DOC_ID)"
