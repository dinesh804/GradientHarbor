"""Hash-chained, HMAC-authenticated and Ed25519-signed JSONL audit events."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
from gradientharbor.security.crypto import hmac_hex, verify_hmac, sign_bytes, verify_signature
GENESIS = "0" * 64

def canonical(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)

def digest(prev: str, payload: dict[str, Any]) -> str:
    return hashlib.sha256((prev + "\n" + canonical(payload)).encode("utf-8")).hexdigest()

def read_events(path: str | Path) -> list[dict[str, Any]]:
    p=Path(path)
    if not p.exists(): return []
    out=[]
    with p.open("r", encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            if line.strip():
                try: out.append(json.loads(line))
                except json.JSONDecodeError as exc: raise ValueError(f"Malformed audit record at line {i}") from exc
    return out

def append_event(path: str | Path, event: dict[str, Any]) -> dict[str, Any]:
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    records=read_events(p); prev=records[-1].get("hash", "") if records else GENESIS
    payload={"event":event,"previous_hash":prev}
    record={**payload,"hash":digest(prev,payload)}
    signed=canonical({"event":event,"previous_hash":prev,"hash":record["hash"]}).encode("utf-8")
    record["hmac_sha256"]=hmac_hex(signed)
    record["ed25519_signature"]=sign_bytes(signed)
    with p.open("a",encoding="utf-8",newline="\n") as f:
        f.write(canonical(record)+"\n"); f.flush()
    return record

def verify_chain(path: str | Path) -> tuple[bool, str]:
    p=Path(path)
    if not p.exists(): return True,"No audit records yet."
    try: records=read_events(p)
    except ValueError as exc: return False,str(exc)
    prev=GENESIS
    for i,rec in enumerate(records,1):
        event=rec.get("event")
        if not isinstance(event,dict): return False,f"Record {i} has invalid event."
        if rec.get("previous_hash") != prev: return False,f"Record {i} previous hash mismatch."
        payload={"event":event,"previous_hash":prev}
        expected=digest(prev,payload)
        if rec.get("hash") != expected: return False,f"Record {i} hash mismatch."
        signed=canonical({"event":event,"previous_hash":prev,"hash":expected}).encode("utf-8")
        if not verify_hmac(signed,rec.get("hmac_sha256","")): return False,f"Record {i} HMAC verification failed."
        if not verify_signature(signed,rec.get("ed25519_signature","")): return False,f"Record {i} Ed25519 signature verification failed."
        prev=expected
    return True,f"Verified {len(records)} audit record(s): hash chain, HMAC and Ed25519 signatures valid."
