from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gradientharbor.security.crypto import ensure_keys
if __name__ == "__main__":
    print("GradientHarbor local development keys:")
    for name, path in ensure_keys().items(): print(f"- {name}: {path}")
    print("Keep .secrets/ private. Do not commit it. Back up keys securely; losing keys makes encrypted documents unrecoverable.")
