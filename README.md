# rag

To activate environment run:

```
poetry shell
```

Whenever adding a new dependency run

```
poetry add <package_name>
```

And make sure lock file gets updated.

## Capstone ingestion

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Then:

```
poetry install          # registers the `ingest` command
poetry run ingest       # load docs, chunk, build Chroma + BM25 indexes
```

Cheap checks that skip the embedding API:

```
poetry run ingest --dry-run --max-pages 5
```

Code layout under `rag/capstone/`:

- `cli.py` — entrypoint: argument parsing and console output only
- `config.py` — `Settings` dataclass, loaded from `.env` and the environment
- `ingestion/pipeline.py` — orchestrates the steps below and returns a report
- `ingestion/loading.py`, `chunking.py`, `indexing.py` — one step each, no I/O beyond their own concern
