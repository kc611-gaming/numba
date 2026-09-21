#!/usr/bin/env python3
"""Build a local SQLite index of every place NumPy is used in the repository.

The index is intended to be queried (by humans or LLMs) with SQL. A JSONL
export is written alongside it for direct reading.

Usage:
    python numpy_usage_db/build.py [repo_root] [out_dir]
"""
from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import sys
import tokenize
from pathlib import Path

SCHEMA = """
DROP TABLE IF EXISTS meta;
DROP TABLE IF EXISTS files;
DROP TABLE IF EXISTS numpy_hits;
DROP VIEW IF EXISTS usage_summary;
DROP VIEW IF EXISTS file_summary;

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE files (
    id        INTEGER PRIMARY KEY,
    path      TEXT NOT NULL UNIQUE,
    ext       TEXT NOT NULL,
    language  TEXT NOT NULL,
    num_lines INTEGER NOT NULL
);

CREATE TABLE numpy_hits (
    id        INTEGER PRIMARY KEY,
    file_id   INTEGER NOT NULL REFERENCES files(id),
    line      INTEGER NOT NULL,
    col       INTEGER NOT NULL,
    kind      TEXT NOT NULL,   -- import | import_alias | from_import | attribute | bare_name | text_ref
    alias     TEXT,            -- e.g. np, nmp
    symbol    TEXT,            -- top-level numpy symbol, e.g. linalg, ndarray
    qualified TEXT,            -- full dotted reference, e.g. np.linalg.solve
    snippet   TEXT NOT NULL
);

CREATE INDEX idx_hits_file     ON numpy_hits(file_id);
CREATE INDEX idx_hits_kind     ON numpy_hits(kind);
CREATE INDEX idx_hits_symbol   ON numpy_hits(symbol);
CREATE INDEX idx_hits_qualified ON numpy_hits(qualified);
CREATE INDEX idx_hits_line     ON numpy_hits(file_id, line);

CREATE VIEW usage_summary AS
SELECT h.kind,
       COALESCE(h.symbol, '') AS symbol,
       COUNT(*)               AS hits,
       COUNT(DISTINCT h.file_id) AS files
FROM numpy_hits AS h
GROUP BY h.kind, h.symbol
ORDER BY hits DESC;

CREATE VIEW file_summary AS
SELECT f.path,
       COUNT(*) AS hits,
       SUM(h.kind IN ('import', 'import_alias', 'from_import')) AS imports
FROM numpy_hits AS h
JOIN files AS f ON f.id = h.file_id
GROUP BY f.id
ORDER BY hits DESC, f.path;
"""

SKIP_DIRS = {
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "node_modules", ".eggs", "build", "dist", ".venv", "venv",
}
SKIP_EXTS = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".dll", ".exe", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".gz", ".bz2", ".xz",
    ".zip", ".tar", ".whl", ".npy", ".npz", ".pickle", ".pkl", ".woff",
    ".woff2", ".ttf", ".eot", ".mp4", ".webp",
}
TEXT_EXTS = {
    ".py", ".pyi", ".pyx", ".pxd", ".rst", ".md", ".txt", ".cfg", ".ini",
    ".toml", ".yaml", ".yml", ".json", ".c", ".h", ".cpp", ".hpp", ".cu",
    ".sh", ".bat", ".cmake", ".ipynb", ".numba", ".pyt", ".csv", ".xtc",
}
LANGUAGE = {
    ".py": "python", ".pyi": "python", ".pyx": "cython", ".pxd": "cython",
    ".rst": "rst", ".md": "markdown", ".ipynb": "notebook", ".numba": "numba",
    ".c": "c", ".h": "c", ".cpp": "c++", ".hpp": "c++", ".cu": "cuda",
    ".sh": "shell", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml",
    ".json": "json", ".ini": "ini", ".cfg": "ini",
    ".txt": "text",
}

IMPORT_RE = re.compile(
    r"^[ \t]*import[ \t]+numpy(?:[ \t]+as[ \t]+([A-Za-z_]\w*))?"
)
FROM_RE = re.compile(
    r"^[ \t]*from[ \t]+numpy(?:\.([A-Za-z_][\w.]*))?[ \t]+import[ \t]+(.*)$"
)
IDENT_RE = re.compile(r"[A-Za-z_]\w*")
TEXT_REF_RE = re.compile(r"\bnumpy\b|(?<![\w.])np\.[A-Za-z_]\w*")


def read_text(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:8192]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def strip_comment(line: str, language: str) -> str:
    if language == "python":
        hash_pos = line.find("#")
        return line if hash_pos < 0 else line[:hash_pos]
    return line


def collect_python_imports(lines: list[str], hits: list[tuple]) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {"numpy": set()}
    bare: set[str] = set()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        m_import = IMPORT_RE.match(raw)
        if m_import:
            alias = m_import.group(1)
            kind = "import_alias" if alias else "import"
            aliases.setdefault(alias or "numpy", set()).add(alias or "numpy")
            hits.append((i + 1, m_import.start() + 1, kind, alias or "numpy",
                         "numpy", "import numpy as %s" % alias if alias else "import numpy",
                         raw.rstrip()))
            i += 1
            continue

        m_from = FROM_RE.match(raw)
        if m_from:
            submodule = m_from.group(1)
            body = m_from.group(2)
            start = i
            while body.count("(") > body.count(")") and i + 1 < len(lines):
                i += 1
                body += " " + lines[i]
            body = strip_comment(body, "python").replace("\\", " ")
            body = body.strip().rstrip(")").lstrip("(")
            names = []
            for part in body.split(","):
                part = part.strip()
                if not part:
                    continue
                name = part.split(" as ")[0].strip()
                if IDENT_RE.fullmatch(name):
                    names.append(name)
                    bare.add(name)
            qualified_base = ("numpy." + submodule) if submodule else "numpy"
            for name in names:
                hits.append((start + 1, m_from.start() + 1, "from_import",
                             None, name, "%s.%s" % (qualified_base, name),
                             lines[start].rstrip()))
            i += 1
            continue
        i += 1
    return {"aliases": set(aliases), "bare": bare}


def scan_python(path: Path, lines: list[str], rel: str) -> list[tuple]:
    """Tokenize the file so strings, comments and docstrings are ignored."""
    hits: list[tuple] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO("\n".join(lines)).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return scan_python_regex(path, lines, rel)

    n = len(tokens)
    aliases: set[str] = {"numpy"}
    bare: set[str] = set()
    import_toks: set[int] = set()

    i = 0
    while i < n:
        tok = tokens[i]
        if tok.type != tokenize.NAME or tok.string not in ("import", "from"):
            i += 1
            continue
        if tok.string == "import":
            j = i + 1
            while j < n and tokens[j].type not in (tokenize.NEWLINE, tokenize.ENDMARKER) \
                    and tokens[j].string != ";":
                t = tokens[j]
                import_toks.add(j)
                if t.type == tokenize.NAME and t.string == "numpy":
                    alias = None
                    if j + 2 < n and tokens[j + 1].string == "as" \
                            and tokens[j + 2].type == tokenize.NAME:
                        alias = tokens[j + 2].string
                        import_toks.update((j + 1, j + 2))
                    aliases.add(alias or "numpy")
                    hits.append((t.start[0], t.start[1] + 1,
                                 "import_alias" if alias else "import", alias,
                                 "numpy", "import numpy as %s" % alias if alias
                                 else "import numpy", lines[t.start[0] - 1].rstrip()))
                j += 1
        else:
            j = i + 1
            if j < n and tokens[j].type == tokenize.NAME and tokens[j].string == "numpy":
                import_toks.add(j)
                k = j + 1
                parts = []
                while k + 1 < n and tokens[k].string == "." \
                        and tokens[k + 1].type == tokenize.NAME:
                    parts.append(tokens[k + 1].string)
                    import_toks.update((k, k + 1))
                    k += 2
                submodule = ".".join(parts)
                if k < n and tokens[k].string == "import":
                    import_toks.add(k)
                    j = k + 1
                    while j < n and tokens[j].type not in (
                            tokenize.NEWLINE, tokenize.ENDMARKER):
                        t = tokens[j]
                        import_toks.add(j)
                        if t.type == tokenize.NAME:
                            alias = None
                            if j + 2 < n and tokens[j + 1].string == "as" \
                                    and tokens[j + 2].type == tokenize.NAME:
                                alias = tokens[j + 2].string
                                import_toks.update((j + 1, j + 2))
                                j += 2
                            bare.add(alias or t.string)
                            base = ("numpy." + submodule) if submodule else "numpy"
                            hits.append((t.start[0], t.start[1] + 1, "from_import",
                                         None, t.string, "%s.%s" % (base, t.string),
                                         lines[t.start[0] - 1].rstrip()))
                        j += 1
        i = j

    for idx, tok in enumerate(tokens):
        if idx in import_toks or tok.type != tokenize.NAME:
            continue
        prev = tokens[idx - 1].string if idx else ""
        if tok.string in aliases and prev != "." \
                and idx + 1 < n and tokens[idx + 1].string == ".":
            j = idx + 1
            parts = []
            while j + 1 < n and tokens[j].string == "." \
                    and tokens[j + 1].type == tokenize.NAME:
                parts.append(tokens[j + 1].string)
                j += 2
            hits.append((tok.start[0], tok.start[1] + 1, "attribute", tok.string,
                         parts[0], "%s.%s" % (tok.string, ".".join(parts)),
                         lines[tok.start[0] - 1].rstrip()))
        elif tok.string in bare and prev != ".":
            hits.append((tok.start[0], tok.start[1] + 1, "bare_name", None,
                         tok.string, tok.string, lines[tok.start[0] - 1].rstrip()))
    return hits


def scan_python_regex(path: Path, lines: list[str], rel: str) -> list[tuple]:
    hits: list[tuple] = []
    info = collect_python_imports(lines, hits)
    aliases = info["aliases"]
    bare = info["bare"]

    usage_res = [
        (alias, re.compile(r"(?<![\w.])%s\.([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)" % re.escape(alias)))
        for alias in sorted(aliases, key=len, reverse=True)
    ]
    bare_res = sorted(bare)
    import_lines = {h[0] for h in hits}

    for lineno, raw in enumerate(lines, start=1):
        code = strip_comment(raw, "python")
        if not code.strip() or lineno in import_lines:
            continue
        found = False
        for alias, rx in usage_res:
            for m in rx.finditer(code):
                chain = m.group(1)
                symbol = chain.split(".")[0]
                hits.append((lineno, m.start() + 1, "attribute", alias, symbol,
                             "%s.%s" % (alias, chain), raw.rstrip()))
                found = True
        if found:
            continue
        for name in bare_res:
            for m in re.finditer(r"(?<![\w.])%s(?![\w])" % re.escape(name), code):
                hits.append((lineno, m.start() + 1, "bare_name", None, name,
                             name, raw.rstrip()))
                break
    return hits


def scan_text(path: Path, lines: list[str], rel: str) -> list[tuple]:
    hits: list[tuple] = []
    for lineno, raw in enumerate(lines, start=1):
        for m in TEXT_REF_RE.finditer(raw):
            token = m.group(0)
            alias, _, rest = token.partition(".")
            symbol = rest.split(".")[0] if rest else "numpy"
            hits.append((lineno, m.start() + 1, "text_ref",
                         alias if alias in ("np",) else None,
                         symbol or "numpy", token, raw.rstrip()))
    return hits


def iter_files(root: Path, out_dir: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            try:
                path.relative_to(out_dir)
                continue
            except ValueError:
                pass
            ext = path.suffix.lower()
            if ext in SKIP_EXTS:
                continue
            if ext and ext not in TEXT_EXTS:
                continue
            yield path


def build(root: Path, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "numpy_usage.db"
    jsonl_path = out_dir / "numpy_usage.jsonl"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO meta VALUES ('repo_root', ?)", (str(root),))

    file_count = hit_count = 0
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for path in sorted(iter_files(root, out_dir)):
            rel = str(path.relative_to(root))
            text = read_text(path)
            if text is None:
                continue
            lines = text.splitlines()
            ext = path.suffix.lower()
            language = LANGUAGE.get(ext, "text")
            hits = (scan_python(path, lines, rel) if language == "python"
                    else scan_text(path, lines, rel))
            if not hits:
                continue
            file_count += 1
            cur = conn.execute(
                "INSERT INTO files(path, ext, language, num_lines) VALUES (?,?,?,?)",
                (rel, ext, language, len(lines)),
            )
            file_id = cur.lastrowid
            rows = [(file_id, ln, col, kind, alias, sym, qual, snip)
                    for ln, col, kind, alias, sym, qual, snip in hits]
            conn.executemany(
                "INSERT INTO numpy_hits(file_id, line, col, kind, alias, symbol,"
                " qualified, snippet) VALUES (?,?,?,?,?,?,?,?)",
                rows,
            )
            hit_count += len(rows)
            for ln, col, kind, alias, sym, qual, snip in hits:
                jf.write(json.dumps({
                    "path": rel, "line": ln, "col": col, "kind": kind,
                    "alias": alias, "symbol": sym, "qualified": qual,
                    "snippet": snip,
                }, ensure_ascii=False) + "\n")

    conn.execute("INSERT INTO meta VALUES ('files_indexed', ?)", (str(file_count),))
    conn.execute("INSERT INTO meta VALUES ('hits', ?)", (str(hit_count),))
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    return db_path, jsonl_path


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    out_dir = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(__file__).resolve().parent
    db_path, jsonl_path = build(root, out_dir)
    conn = sqlite3.connect(db_path)
    meta = dict(conn.execute("SELECT key, value FROM meta"))
    conn.close()
    print("root:     %s" % root)
    print("database: %s" % db_path)
    print("jsonl:    %s" % jsonl_path)
    print("files with numpy usage: %s" % meta.get("files_indexed"))
    print("numpy hits:             %s" % meta.get("hits"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
