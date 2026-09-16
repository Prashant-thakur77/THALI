"""Hash-chained audit log (plan Phase 5.2).

Every record is one JSON line: {"seq", "ts", "kind", "payload", "prev_hash", "sha256"} where sha256 is
computed over the canonical JSON (sorted keys, no whitespace) of the record *without* its own sha256 field,
and prev_hash is the previous record's sha256 (the genesis record chains to 64 zeros).  Editing, deleting or
re-ordering any line breaks every later hash; ``verify`` recomputes the chain and reports the first bad line.
``make verify-log`` runs ``python -m verifier.audit verify <file>``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def record_hash(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "sha256"}
    return hashlib.sha256(canonical(body)).hexdigest()


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seq = 0
        self.prev_hash = GENESIS
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.seq = rec["seq"] + 1
                    self.prev_hash = rec["sha256"]

    def append(self, kind: str, payload: dict) -> dict:
        rec = {"seq": self.seq, "ts": round(time.time(), 3), "kind": kind, "payload": payload, "prev_hash": self.prev_hash}
        rec["sha256"] = record_hash(rec)
        with self.path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        self.seq += 1
        self.prev_hash = rec["sha256"]
        return rec


def verify(path: Path) -> dict:
    """Recompute the chain. Returns {"ok", "records", "first_bad_seq", "reason"}."""
    prev = GENESIS
    n = 0
    for lineno, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("seq") != n:
            return {"ok": False, "records": n, "first_bad_seq": n, "reason": f"line {lineno}: seq {rec.get('seq')} != {n} (deleted/reordered)"}
        if rec.get("prev_hash") != prev:
            return {"ok": False, "records": n, "first_bad_seq": n, "reason": f"line {lineno}: prev_hash mismatch"}
        if record_hash(rec) != rec.get("sha256"):
            return {"ok": False, "records": n, "first_bad_seq": n, "reason": f"line {lineno}: sha256 mismatch (edited)"}
        prev = rec["sha256"]
        n += 1
    return {"ok": True, "records": n, "first_bad_seq": None, "reason": "chain intact"}


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "verify":
        res = verify(Path(sys.argv[2]))
        print(json.dumps(res))
        sys.exit(0 if res["ok"] else 1)
    print("usage: python -m verifier.audit verify <audit.jsonl>")
    sys.exit(2)
