from __future__ import annotations
from pathlib import Path
import json, re
from typing import Any
from gradientharbor.security.integrity import sha256_bytes, sign_digest, verify_digest_signature, decrypt_text
from gradientharbor.security.prompt_guard import scan_untrusted_text
BASE=Path(__file__).resolve().parents[2]
KB_DIR=BASE/"knowledge_base"
MANIFEST_PATH=KB_DIR/".integrity_manifest.json"

def _load_manifest():
    try: return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError,json.JSONDecodeError): return {}
def _save_manifest(manifest):
    KB_DIR.mkdir(parents=True,exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding="utf-8")
def _read_doc(path: Path) -> str:
    return decrypt_text(path.read_bytes()) if path.suffix.lower()==".enc" else path.read_text(encoding="utf-8")
def _source_name(path: Path) -> str:
    return path.name[:-4] if path.name.endswith(".enc") else path.name

def ingest_text(source: str, content: str, preserve_existing: bool = True):
    manifest=_load_manifest(); digest=sha256_bytes(content.encode("utf-8")); old=manifest.get(source)
    if isinstance(old,str): old={"sha256":old}
    if not preserve_existing or source not in manifest:
        manifest[source]={"sha256":digest,"signature":sign_digest(digest)}
        _save_manifest(manifest)
    return {"source":source,"sha256":digest,"signature_valid":verify_digest_signature(digest,manifest[source].get("signature","")) if isinstance(manifest.get(source),dict) else False}

def _split(text: str, size: int = 900):
    text=re.sub(r"\s+"," ",text).strip()
    if not text:return []
    parts=re.split(r"(?<=[.!?])\s+",text); chunks=[]; current=""
    for part in parts:
        if current and len(current)+len(part)+1>size: chunks.append(current); current=part
        else: current=(current+" "+part).strip()
    if current:chunks.append(current)
    return chunks

def _docs():
    KB_DIR.mkdir(parents=True,exist_ok=True); manifest=_load_manifest(); changed=False; out=[]
    for p in sorted(KB_DIR.iterdir()):
        if not p.is_file() or not (p.suffix.lower() in (".md",".txt") or p.name.endswith((".md.enc",".txt.enc"))): continue
        try: content=_read_doc(p)
        except Exception: continue
        source=_source_name(p); digest=sha256_bytes(content.encode("utf-8")); entry=manifest.get(source)
        # Never sign a newly discovered file during retrieval: that would allow an
        # untrusted drop-in document to self-promote to trusted evidence.
        # Only the explicit ingestion workflow signs evidence.
        signature_ok=isinstance(entry,dict) and entry.get("sha256")==digest and verify_digest_signature(digest,entry.get("signature",""))
        for i,chunk in enumerate(_split(content)):
            scan=scan_untrusted_text(chunk)
            out.append({"source":source,"chunk":i,"text":chunk,"sha256":digest,
                        "signature_valid":signature_ok,"injection_risk":scan["suspicious"],"injection_signals":scan["signals"]})
    if changed:_save_manifest(manifest)
    return out

def retrieve(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    docs=_docs()
    if not query.strip() or not docs:return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec=TfidfVectorizer(stop_words="english"); mat=vec.fit_transform([d["text"] for d in docs]+[query])
        scores=(mat[-1]@mat[:-1].T).toarray().ravel(); ranked=sorted(zip(scores,docs),key=lambda x:x[0],reverse=True)
        return [{**d,"score":float(s)} for s,d in ranked[:top_k] if s>0]
    except Exception:
        terms=set(re.findall(r"\w+",query.lower())); ranked=[]
        for d in docs:
            dt=set(re.findall(r"\w+",d["text"].lower())); score=len(terms&dt)/max(1,len(terms))
            if score:ranked.append((score,d))
        ranked.sort(key=lambda x:x[0],reverse=True)
        return [{**d,"score":float(s)} for s,d in ranked[:top_k]]

def verify_documents():
    docs=_docs(); by_source={}
    for d in docs: by_source[d["source"]]=d
    out=[]
    for source,d in sorted(by_source.items()):
        out.append({"source":source,"sha256":d["sha256"],"signature_valid":d["signature_valid"],"valid":d["signature_valid"]})
    return out
