from __future__ import annotations
import hashlib
from gradientharbor.security.crypto import sign_bytes, verify_signature, encrypt_bytes, decrypt_bytes

def sha256_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def verify_bytes(data: bytes, expected_sha256: str) -> bool: return hashlib.sha256(data).hexdigest() == expected_sha256
def sign_digest(digest: str) -> str: return sign_bytes(digest.encode("ascii"))
def verify_digest_signature(digest: str, signature: str) -> bool: return verify_signature(digest.encode("ascii"), signature)
def encrypt_text(text: str) -> bytes: return encrypt_bytes(text.encode("utf-8"))
def decrypt_text(data: bytes) -> str: return decrypt_bytes(data).decode("utf-8")
