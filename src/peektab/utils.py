from __future__ import annotations
import os
import sys
from typing import Iterable, Optional, Tuple

import polars as pl
from rich.table import Table
from rich.console import Console
from rich import box

console = Console()

# -------- File loading --------
def detect_format(path: str, fmt: Optional[str]) -> str:
    if fmt:
        return fmt.lower()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        return "csv"
    if ext in (".jsonl", ".ndjson"):
        return "ndjson"
    if ext in (".parquet", ".pq"):
        return "parquet"
    # fallback try csv
    return "csv"

def read_frame(path: str, fmt: Optional[str] = None, delimiter: Optional[str] = None, infer_schema_length: int = 2048) -> pl.DataFrame:
    fmt = detect_format(path, fmt)
    if fmt == "csv":
        # auto-delimiter if not provided
        if delimiter is None:
            delimiter = sniff_delimiter(path)
        return pl.read_csv(
            path,
            separator=delimiter,
            infer_schema_length=infer_schema_length,
            try_parse_dates=True
        )
    elif fmt == "ndjson":
        return pl.read_ndjson(path)
    elif fmt == "parquet":
        return pl.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

def scan_lazy(path: str, fmt: Optional[str] = None, delimiter: Optional[str] = None) -> pl.LazyFrame:
    fmt = detect_format(path, fmt)
    if fmt == "csv":
        if delimiter is None:
            delimiter = sniff_delimiter(path)
        return pl.scan_csv(path, separator=delimiter, try_parse_dates=True)
    elif fmt == "ndjson":
        return pl.scan_ndjson(path)
    elif fmt == "parquet":
        return pl.scan_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

def sniff_delimiter(path: str) -> str:
    # very small heuristic: look at first non-empty line
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue
            counts = {",": line.count(","), "\t": line.count("\t"), ";": line.count(";")}
            return max(counts, key=counts.get) if any(counts.values()) else ","
    return ","

# -------- Rendering --------
def render_table(df: pl.DataFrame, max_rows: int = 20, max_width: int = 120, title: Optional[str] = None):
    if df.height > max_rows:
        df = df.head(max_rows)

    tbl = Table(
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        expand=False,
        title=title,
        pad_edge=False
    )
    for col in df.columns:
        tbl.add_column(str(col), overflow="fold", no_wrap=False, justify="left")

    # Convert to strings to avoid Rich complaining about certain dtypes
    for row in df.iter_rows():
        tbl.add_row(*[repr_cell(x) for x in row])

    console.print(tbl, width=max_width)

def repr_cell(val) -> str:
    if val is None:
        return "∅"
    if isinstance(val, float):
        return f"{val:.6g}"
    if isinstance(val, (list, tuple, set, dict)):
        s = str(val)
        return s if len(s) <= 80 else s[:77] + "..."
    return str(val)

def print_kv(title: str, pairs: Iterable[Tuple[str, str]]):
    console.rule(f"[bold]{title}[/bold]")
    t = Table(box=box.SIMPLE, show_lines=False, show_header=False)
    t.add_column("k", style="bold cyan", no_wrap=True)
    t.add_column("v")
    for k, v in pairs:
        t.add_row(k, v)
    console.print(t)

NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}

try:
    NUMERIC_DTYPES.add(pl.Decimal)  # dtype comparison works by base type
except Exception:
    pass
    
def is_numeric_dtype(dt) -> bool:
    # handle Decimal with precision/scale where direct equality may differ
    return (dt in NUMERIC_DTYPES) or ("Decimal" in repr(dt))

def is_utf8_dtype(dt) -> bool:
    return dt == pl.Utf8
