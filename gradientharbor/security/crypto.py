"""Local cryptographic primitives for GradientHarbor development/demo deployments.

Keys are generated locally on first use under .secrets/. Protect and back up these
keys outside the application host before relying on them. This is not an HSM/KMS.
"""
from __future__ import annotations
import base64, hashlib, hmac, os
from pathlib import Path
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

ROOT = Path(__file__).resolve().parents[2]
KEY_DIR = ROOT / ".secrets"
PRIVATE_KEY = KEY_DIR / "ed25519_private.pem"
PUBLIC_KEY = KEY_DIR / "ed25519_public.pem"
FERNET_KEY = KEY_DIR / "fernet.key"
HMAC_KEY = KEY_DIR / "hmac.key"

class CryptoConfigurationError(RuntimeError): pass

def _write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    try: path.chmod(0o600)
    except OSError: pass

def ensure_keys() -> dict[str, str]:
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    known = [PRIVATE_KEY, PUBLIC_KEY, FERNET_KEY, HMAC_KEY]
    existing = [p.exists() for p in known]
    if any(existing) and not all(existing):
        raise CryptoConfigurationError(
            "GradientHarbor key set is incomplete. Restore the missing key from your secure backup; "
            "do not silently generate replacement keys for existing encrypted data or signatures."
        )
    if not any(existing):
        private = Ed25519PrivateKey.generate()
        _write_private(PRIVATE_KEY, private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()))
        _write_private(PUBLIC_KEY, private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo))
        _write_private(FERNET_KEY, Fernet.generate_key())
        _write_private(HMAC_KEY, base64.urlsafe_b64encode(os.urandom(32)))
    return {"signing": str(PRIVATE_KEY), "verification": str(PUBLIC_KEY), "encryption": str(FERNET_KEY), "hmac": str(HMAC_KEY)}

def _hmac_key() -> bytes:
    env = os.getenv("GRADIENTHARBOR_HMAC_KEY", "").strip()
    if env: return env.encode("utf-8")
    ensure_keys()
    return base64.urlsafe_b64decode(HMAC_KEY.read_bytes())

def hmac_hex(data: bytes) -> str:
    return hmac.new(_hmac_key(), data, hashlib.sha256).hexdigest()

def verify_hmac(data: bytes, expected: str) -> bool:
    return hmac.compare_digest(hmac_hex(data), expected or "")

def sign_bytes(data: bytes) -> str:
    ensure_keys()
    key = serialization.load_pem_private_key(PRIVATE_KEY.read_bytes(), password=None)
    return base64.b64encode(key.sign(data)).decode("ascii")

def verify_signature(data: bytes, signature: str) -> bool:
    try:
        ensure_keys()
        key = serialization.load_pem_public_key(PUBLIC_KEY.read_bytes())
        key.verify(base64.b64decode(signature.encode("ascii")), data)
        return True
    except Exception:
        return False

def encrypt_bytes(data: bytes) -> bytes:
    ensure_keys()
    return Fernet(FERNET_KEY.read_bytes()).encrypt(data)

def decrypt_bytes(data: bytes) -> bytes:
    ensure_keys()
    try: return Fernet(FERNET_KEY.read_bytes()).decrypt(data)
    except InvalidToken as exc: raise CryptoConfigurationError("Encrypted content could not be decrypted: key mismatch or ciphertext tampering.") from exc

def key_status() -> dict[str, bool]:
    return {"Ed25519 signing key": PRIVATE_KEY.exists(), "Ed25519 public key": PUBLIC_KEY.exists(),
            "Fernet encryption key": FERNET_KEY.exists(), "HMAC key": HMAC_KEY.exists()}
