from __future__ import annotations

import os
import sys
from typing import Optional, List

import polars as pl
import typer
from rich.console import Console
from rich.panel import Panel
from rich import box

from .utils import (
    read_frame, scan_lazy, detect_format, sniff_delimiter,
    render_table, print_kv,
    is_numeric_dtype, is_utf8_dtype
)

app = typer.Typer(add_completion=False, help="peektab: inspect, summarize, and convert data in your terminal")
console = Console()

# ------------------ Commands ------------------

@app.command(help="Show a neat, truncated preview of the data (head).")
def show(
    path: str = typer.Argument(..., help="Path to CSV/TSV, NDJSON (jsonl), or Parquet file."),
    rows: int = typer.Option(20, "--rows", "-n", help="Number of rows to display."),
    cols: Optional[str] = typer.Option(None, "--cols", "-c", help="Comma-separated list of columns to select."),
    fmt: Optional[str] = typer.Option(None, "--format", "-f", help="Force format: csv|ndjson|parquet"),
    delimiter: Optional[str] = typer.Option(None, "--delimiter", "-d", help="CSV delimiter if not comma."),
):
    lf = scan_lazy(path, fmt=fmt, delimiter=delimiter)
    if cols:
        selected = [c.strip() for c in cols.split(",") if c.strip()]
        lf = lf.select(selected)
    df = lf.fetch(rows)
    title = f"[bold]Preview[/bold] • {os.path.basename(path)}"
    render_table(df, max_rows=rows, title=title)
    console.print(f"[dim]rows shown: {df.height} (file preview) | columns: {len(df.columns)}[/dim]")

@app.command(help="Print inferred schema with dtypes and null counts.")
def schema(
    path: str = typer.Argument(...),
    fmt: Optional[str] = typer.Option(None, "--format", "-f"),
    delimiter: Optional[str] = typer.Option(None, "--delimiter", "-d")
):
    df = read_frame(path, fmt=fmt, delimiter=delimiter, infer_schema_length=10_000)
    null_df = df.null_count()
    # row(0) returns a tuple of counts aligned to null_df.columns
    null_counts = dict(zip(null_df.columns, null_df.row(0)))
    meta = []
    for name, dtype in zip(df.columns, df.dtypes):
        meta.append((name, f"{dtype} · nulls={int(null_counts.get(name, 0))}"))
    print_kv(f"Schema • {os.path.basename(path)}", meta)

@app.command(help="Compute quick stats (numeric summary + top categories).")
def stats(
    path: str = typer.Argument(...),
    topk: int = typer.Option(5, "--topk", help="Top-k categories for string/categorical columns."),
    fmt: Optional[str] = typer.Option(None, "--format", "-f"),
    delimiter: Optional[str] = typer.Option(None, "--delimiter", "-d")
):
    df = read_frame(path, fmt=fmt, delimiter=delimiter)

    # Numeric summary
    num_cols = [c for c, dt in zip(df.columns, df.dtypes) if is_numeric_dtype(dt)]
    if num_cols:
        summary = df.select([
            pl.len().alias("count"),
            *[pl.col(c).mean().alias(f"{c}_mean") for c in num_cols],
            *[pl.col(c).std().alias(f"{c}_std") for c in num_cols],
            *[pl.col(c).min().alias(f"{c}_min") for c in num_cols],
            *[pl.col(c).max().alias(f"{c}_max") for c in num_cols],
        ])
        render_table(summary, title="[bold]Numeric summary[/bold]")
    else:
        console.print(Panel.fit("No numeric columns detected.", border_style="yellow"))

    # Categorical peek (most frequent values)
    str_cols = [c for c, dt in zip(df.columns, df.dtypes) if is_utf8_dtype(dt)]
    for c in str_cols:
        freq = (
            df.select(pl.col(c))
              .drop_nulls()
              .group_by(c)                 # <-- correct API
              .len()
              .sort("len", descending=True)
              .head(topk)
        )
        if freq.height:
            render_table(freq, title=f"[bold]Top {topk} values[/bold] • {c}")

@app.command(help="Sample random rows.")
def sample(
    path: str = typer.Argument(...),
    n: int = typer.Option(10, "--n"),
    seed: Optional[int] = typer.Option(42, "--seed"),
    fmt: Optional[str] = typer.Option(None, "--format", "-f"),
    delimiter: Optional[str] = typer.Option(None, "--delimiter", "-d"),
):
    df = read_frame(path, fmt=fmt, delimiter=delimiter)
    n = min(n, df.height) if df.height else 0
    if n == 0:
        console.print(Panel.fit("Empty dataset.", border_style="red"))
        raise typer.Exit(1)
    s = df.sample(n=n, seed=seed) if seed is not None else df.sample(n=n)
    render_table(s, max_rows=n, title=f"[bold]Random sample ({n})[/bold] • {os.path.basename(path)}")

@app.command(help="List columns.")
def columns(
    path: str = typer.Argument(...),
    fmt: Optional[str] = typer.Option(None, "--format", "-f"),
    delimiter: Optional[str] = typer.Option(None, "--delimiter", "-d")
):
    df = read_frame(path, fmt=fmt, delimiter=delimiter, infer_schema_length=10_000)
    pairs = [(c, str(dt)) for c, dt in zip(df.columns, df.dtypes)]
    print_kv(f"Columns • {os.path.basename(path)}", pairs)

@app.command(help="Convert between formats (csv <-> parquet <-> ndjson).")
def convert(
    src: str = typer.Argument(..., help="Source file"),
    dst: str = typer.Argument(..., help="Destination file (.csv/.parquet/.jsonl)"),
    fmt_src: Optional[str] = typer.Option(None, "--from-format", "-F", help="csv|parquet|ndjson"),
    fmt_dst: Optional[str] = typer.Option(None, "--to-format", "-T", help="csv|parquet|ndjson"),
    delimiter: Optional[str] = typer.Option(None, "--delimiter", "-d", help="Delimiter for CSV output, default comma"),
):
    df = read_frame(src, fmt=fmt_src)
    out_fmt = fmt_dst or os.path.splitext(dst)[1].lower().lstrip(".")
    out_fmt = {"jsonl": "ndjson", "pq": "parquet"}.get(out_fmt, out_fmt)

    if out_fmt == "csv":
        sep = delimiter or ","
        df.write_csv(dst, separator=sep)
    elif out_fmt == "parquet":
        df.write_parquet(dst)
    elif out_fmt == "ndjson":
        df.write_ndjson(dst)
    else:
        console.print(f"[red]Unsupported destination format: {out_fmt}[/red]")
        raise typer.Exit(2)
    console.print(f"[green]Wrote[/green] {dst}")

@app.command(help="Quick info: rows, columns, file format, delimiter (if CSV), and memory footprint.")
def info(
    path: str = typer.Argument(...),
    fmt: Optional[str] = typer.Option(None, "--format", "-f"),
    delimiter: Optional[str] = typer.Option(None, "--delimiter", "-d")
):
    actual_fmt = detect_format(path, fmt)
    used_delim = delimiter
    if actual_fmt == "csv" and used_delim is None:
        used_delim = sniff_delimiter(path)
    lf = scan_lazy(path, fmt=fmt, delimiter=used_delim)
    df = lf.fetch(0)  # get columns fast
    n = lf.select(pl.len()).collect().item()
    mem_est = "--"
    try:
        mem_est = f"{sum([c.estimated_size() for c in df.iter_columns()]) / (1024**2):.2f} MiB (columns only)"
    except Exception:
        pass
    pairs = [
        ("file", os.path.basename(path)),
        ("format", actual_fmt),
        ("delimiter", used_delim or "—"),
        ("rows", str(n)),
        ("columns", str(len(df.columns))),
        ("mem_est", mem_est),
    ]
    print_kv("Info", pairs)

if __name__ == "__main__":
    app()
