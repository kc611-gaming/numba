#!/usr/bin/env python3
"""Query the NumPy usage index built by build.py.

Examples:
    python numpy_usage_db/query.py "SELECT * FROM usage_summary LIMIT 20"
    python numpy_usage_db/query.py --schema
    python numpy_usage_db/query.py --file numba/np/arraymath.py
    python numpy_usage_db/query.py "SELECT * FROM file_summary"
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent / "numpy_usage.db"


def run(sql: str, db: Path, limit: int) -> None:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(limit)
        if cur.description is None:
            print("ok")
            return
        cols = [d[0] for d in cur.description]
        print("\t".join(cols))
        for row in rows:
            print("\t".join("" if v is None else str(v) for v in row))
        print("(%d rows shown)" % len(rows))
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sql", nargs="?", help="SQL statement to execute")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--schema", action="store_true", help="print the DDL")
    ap.add_argument("--file", help="show every numpy hit in one file")
    args = ap.parse_args()

    db = Path(args.db)
    if args.schema:
        run("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL "
            "AND name NOT LIKE 'sqlite_%' ORDER BY type, name", db, 1000)
    elif args.file:
        run_file(db, args.file, args.limit)
    elif args.sql:
        run(args.sql, db, args.limit)
    else:
        ap.print_help()
    return 0


def run_file(db: Path, path: str, limit: int) -> None:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT h.line, h.col, h.kind, h.qualified, h.snippet "
        "FROM numpy_hits h JOIN files f ON f.id = h.file_id "
        "WHERE f.path = ? ORDER BY h.line, h.col LIMIT ?", (path, limit)).fetchall()
    conn.close()
    for row in rows:
        print("%5d:%-4d %-10s %-30s %s" % (
            row["line"], row["col"], row["kind"], row["qualified"] or "", row["snippet"]))
    print("(%d rows shown)" % len(rows))


if __name__ == "__main__":
    raise SystemExit(main())
