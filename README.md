# peektab

A fast, friendly CLI to **peek** at tabular data right in your terminal.

- ✅ CSV/TSV, **NDJSON** (jsonl), **Parquet**
- ✅ Clean, readable **tables** in the console
- ✅ **Schema** (types + null counts), **stats**, **sample**
- ✅ **Convert** between csv/parquet/jsonl
- ✅ Auto-detect **delimiter** (csv)

> Built with [Polars](https://www.pola.rs/), [Typer](https://typer.tiangolo.com/), and [Rich](https://rich.readthedocs.io/).

## Install (editable for dev)
```bash
# from the repo root
pip install -e .
# or with pipx
pipx install .
