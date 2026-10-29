from pathlib import Path

from gradientharbor.multimodal import load_media, store_media, verify_media


def test_media_is_encrypted_signed_and_verifiable(tmp_path: Path):
    original = b"gradientharbor test media payload"
    record = store_media(original, "sample.txt", "text/plain", media_dir=tmp_path)
    encrypted_path = tmp_path / record["storage_key"]
    assert encrypted_path.exists()
    assert original not in encrypted_path.read_bytes()
    metadata, recovered = load_media(record["asset_id"], media_dir=tmp_path)
    assert metadata["filename"] == "sample.txt"
    assert recovered == original
    result = verify_media(tmp_path)
    assert len(result) == 1
    assert result[0]["signature_valid"] is True
    assert result[0]["file_integrity_valid"] is True
    assert result[0]["status"] == "valid"


def test_tampered_media_is_detected(tmp_path: Path):
    record = store_media(b"original", "sample.bin", "application/octet-stream", media_dir=tmp_path)
    (tmp_path / record["storage_key"]).write_bytes(b"tampered")
    result = verify_media(tmp_path)
    assert result[0]["status"] == "ALERT"
    assert result[0]["file_integrity_valid"] is False
