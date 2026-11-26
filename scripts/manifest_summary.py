"""Summarize data/raw/manifest.json: files, rows and bytes per stage (and optionally per file)."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voldesk import config  # noqa: E402


def main() -> None:
    man = json.loads((config.RAW / "manifest.json").read_text(encoding="utf-8"))
    agg = collections.defaultdict(lambda: {"files": 0, "rows": 0, "bytes": 0})
    for rec in man.values():
        a = agg[rec["stage"]]
        a["files"] += 1
        a["rows"] += rec["rows"]
        a["bytes"] += rec["bytes"]
    for stage, a in agg.items():
        print(f"{stage:10s} files={a['files']:4d} rows={a['rows']:>13,} MB={a['bytes'] / 1e6:9.1f}")
    print(f"{'total':10s} files={len(man):4d} rows={sum(r['rows'] for r in man.values()):>13,} "
          f"MB={sum(r['bytes'] for r in man.values()) / 1e6:9.1f}")
    if "--files" in sys.argv:
        for k, r in sorted(man.items()):
            print(f"{k:55s} {r['rows']:>11,} {r['retrieved_at_utc']}")


if __name__ == "__main__":
    main()
