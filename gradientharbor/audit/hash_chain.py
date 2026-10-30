"""Hash-chained, HMAC-authenticated and Ed25519-signed JSONL audit events.

Writers serialize the entire read/validate/append sequence with a sidecar file lock,
so concurrent processes cannot append entries that share the same previous hash.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from gradientharbor.security.crypto import hmac_hex, verify_hmac, sign_bytes, verify_signature

GENESIS = "0" * 64
_PROCESS_LOCK = threading.Lock()


def canonical(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def digest(prev: str, payload: dict[str, Any]) -> str:
    return hashlib.sha256((prev + "\n" + canonical(payload)).encode("utf-8")).hexdigest()


@contextmanager
def _exclusive_log_lock(path: Path) -> Iterator[None]:
    """Lock a sidecar file across threads and processes on Windows and POSIX."""
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _PROCESS_LOCK:
        with lock_path.open("a+b") as handle:
            if os.name == "nt":
                import msvcrt
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_events(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed audit record at line {line_number}") from exc
    return out


def _verify_records(records: list[dict[str, Any]]) -> tuple[bool, str]:
    previous = GENESIS
    for index, record in enumerate(records, 1):
        event = record.get("event")
        if not isinstance(event, dict):
            return False, f"Record {index} has invalid event."
        if record.get("previous_hash") != previous:
            return False, f"Record {index} previous hash mismatch."
        expected = digest(previous, {"event": event, "previous_hash": previous})
        if record.get("hash") != expected:
            return False, f"Record {index} hash mismatch."
        signed = canonical({"event": event, "previous_hash": previous, "hash": expected}).encode("utf-8")
        if not verify_hmac(signed, record.get("hmac_sha256", "")):
            return False, f"Record {index} HMAC verification failed."
        if not verify_signature(signed, record.get("ed25519_signature", "")):
            return False, f"Record {index} Ed25519 signature verification failed."
        previous = expected
    return True, f"Verified {len(records)} audit record(s): hash chain, HMAC and Ed25519 signatures valid."


def append_event(path: str | Path, event: dict[str, Any]) -> dict[str, Any]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_log_lock(p):
        records = read_events(p)
        valid, reason = _verify_records(records)
        if not valid:
            raise ValueError(f"Refusing to append to an invalid audit chain: {reason}")
        previous = records[-1].get("hash", "") if records else GENESIS
        payload = {"event": event, "previous_hash": previous}
        record = {**payload, "hash": digest(previous, payload)}
        signed = canonical({"event": event, "previous_hash": previous, "hash": record["hash"]}).encode("utf-8")
        record["hmac_sha256"] = hmac_hex(signed)
        record["ed25519_signature"] = sign_bytes(signed)
        with p.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record


def verify_chain(path: str | Path) -> tuple[bool, str]:
    p = Path(path)
    if not p.exists():
        return True, "No audit records yet."
    try:
        with _exclusive_log_lock(p):
            if not p.exists():
                return True, "No audit records yet."
            records = read_events(p)
            if not records:
                return True, "No audit records yet."
            return _verify_records(records)
    except ValueError as exc:
        return False, str(exc)
