"""WRDS access: connection through libpq's password file, COPY-based pulls straight to parquet, and a manifest.

Authentication. Log in once with the official route (the `wrds` package's first connection, or any tool that writes
the standard PostgreSQL password file: %APPDATA%\\postgresql\\pgpass.conf on Windows, ~/.pgpass elsewhere). This module
never asks for, reads or stores a password: libpq reads the file itself. The username comes from WRDS_USERNAME or from
the user field of that file's WRDS line.

Pulls. ``copy_to_parquet`` streams ``COPY (query) TO STDOUT`` into a temporary CSV on disk, converts it block by block
with pyarrow into a zstd parquet file with an explicit schema (no type inference across blocks), and replaces the
destination atomically. ``Manifest`` (data/raw/manifest.json) records, per file: the SQL, its hash, row count, columns,
byte size, sha256 of the parquet and the UTC retrieval time. A rerun skips files whose SQL hash is already recorded.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

HOST, PORT, DB = "wrds-pgdata.wharton.upenn.edu", 9737, "wrds"


def pgpass_path() -> Path:
    if os.name == "nt":
        return Path(os.environ["APPDATA"]) / "postgresql" / "pgpass.conf"
    return Path.home() / ".pgpass"


def wrds_username() -> str | None:
    """WRDS_USERNAME, else the user field of the WRDS line of the libpq password file (the password is not read)."""
    if os.environ.get("WRDS_USERNAME"):
        return os.environ["WRDS_USERNAME"]
    p = pgpass_path()
    if not p.exists():
        return None
    prefix = f"{HOST}:{PORT}:{DB}:"
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.startswith(prefix):
            return ln[len(prefix):].split(":", 1)[0]
    return None


def engine(statement_timeout_ms: int = 3_600_000):
    from sqlalchemy import create_engine
    user = wrds_username()
    if not user:
        raise RuntimeError("No WRDS login found. Log in once so that libpq's password file holds the WRDS line, "
                           "or set WRDS_USERNAME.")
    return create_engine(f"postgresql+psycopg2://{user}@{HOST}:{PORT}/{DB}",
                         connect_args={"sslmode": "require", "connect_timeout": 120,
                                       "options": f"-c statement_timeout={int(statement_timeout_ms)}"},
                         pool_pre_ping=True)


def sql_hash(sql: str) -> str:
    return hashlib.sha256(" ".join(sql.split()).encode("utf-8")).hexdigest()[:16]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Manifest:
    """data/raw/manifest.json: one record per pulled file, keyed by its path relative to the raw root."""

    def __init__(self, raw_root: Path):
        self.root = Path(raw_root)
        self.path = self.root / "manifest.json"
        self.records: dict = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def has(self, rel: str, sql: str) -> bool:
        rec = self.records.get(rel)
        return bool(rec) and rec.get("sql_hash") == sql_hash(sql) and (self.root / rel).exists()

    def add(self, rel: str, rec: dict) -> None:
        self.records[rel] = rec
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.records, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)


@dataclass
class PullSpec:
    rel: str                                 # destination relative to the raw root, e.g. optionm/opprcd/spx/2024.parquet
    sql: str
    schema: dict[str, pa.DataType]           # every selected column, in order
    stage: str = "misc"
    note: str = ""
    meta: dict = field(default_factory=dict)


def copy_to_parquet(eng, spec: PullSpec, raw_root: Path, tmp_dir: Path | None = None) -> dict:
    """Run one spec: COPY to a temporary CSV, convert to parquet with the spec's schema, replace atomically."""
    raw_root = Path(raw_root)
    dest = raw_root / spec.rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tmp_dir or raw_root / "_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    stem = spec.rel.replace("/", "__").replace(".parquet", "")
    tmp_csv, tmp_pq = tmp_dir / f"{stem}.csv", tmp_dir / f"{stem}.parquet"
    t0 = time.time()
    raw = eng.raw_connection()
    try:
        cur = raw.cursor()
        try:
            with open(tmp_csv, "wb") as f:
                cur.copy_expert(f"COPY ({spec.sql}) TO STDOUT WITH (FORMAT csv, HEADER true)", f)
        finally:
            cur.close()
    finally:
        raw.close()
    t_copy = time.time() - t0
    names = list(spec.schema)
    convert = pacsv.ConvertOptions(column_types=spec.schema, null_values=[""], strings_can_be_null=True,
                                   quoted_strings_can_be_null=False, timestamp_parsers=["%Y-%m-%d"],
                                   true_values=["t", "true", "True"], false_values=["f", "false", "False"])
    n = 0
    schema = pa.schema([(c, spec.schema[c]) for c in names])
    # Open the CSV ourselves so the handle is closed before the file is removed (Windows keeps open files locked).
    with open(tmp_csv, "rb") as fh:
        reader = pacsv.open_csv(fh, read_options=pacsv.ReadOptions(block_size=64 << 20), convert_options=convert)
        got = reader.schema.names
        if got != names:
            raise ValueError(f"{spec.rel}: columns {got} differ from the spec {names}")
        with pq.ParquetWriter(tmp_pq, schema, compression="zstd", compression_level=6) as w:
            for batch in reader:
                w.write_batch(batch.cast(schema) if batch.schema != schema else batch)
                n += batch.num_rows
        del reader
    os.replace(tmp_pq, dest)
    csv_bytes = tmp_csv.stat().st_size
    os.remove(tmp_csv)
    return {"sql": " ".join(spec.sql.split()), "sql_hash": sql_hash(spec.sql), "rows": n, "columns": names,
            "bytes": dest.stat().st_size, "csv_bytes": csv_bytes, "sha256": file_sha256(dest),
            "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seconds_copy": round(t_copy, 1), "seconds_total": round(time.time() - t0, 1),
            "stage": spec.stage, "note": spec.note, **({"meta": spec.meta} if spec.meta else {})}


def read_sql(eng, sql: str):
    """Small metadata queries (pandas)."""
    import pandas as pd
    from sqlalchemy import text
    with eng.connect() as c:
        return pd.read_sql(text(sql), c)
