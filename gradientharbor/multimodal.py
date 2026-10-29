"""Optional multimodal adapters and encrypted media vault for GradientHarbor.

Heavy ML dependencies are imported lazily so the core Nemotron app still starts
without installing local VLM/audio/diffusion packages.
"""
from __future__ import annotations

import hashlib
import io
import json
import mimetypes
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gradientharbor.security.crypto import decrypt_bytes, encrypt_bytes, sign_bytes, verify_signature
from gradientharbor.security.integrity import sha256_bytes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEDIA_DIR = ROOT / "media_store"
MANIFEST_NAME = "manifest.jsonl"


def _canonical(record: dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def store_media(data: bytes, filename: str, mime_type: str | None = None,
                media_dir: str | Path = DEFAULT_MEDIA_DIR, category: str = "upload") -> dict[str, Any]:
    """Encrypt original bytes, sign metadata, and append a local manifest record."""
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > 100 * 1024 * 1024:
        raise ValueError("File exceeds the 100 MB demo upload limit.")
    directory = Path(media_dir)
    directory.mkdir(parents=True, exist_ok=True)
    asset_id = uuid.uuid4().hex
    safe_name = Path(filename).name[:180] or "upload.bin"
    resolved_mime = mime_type or mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
    encrypted_name = f"{asset_id}.enc"
    encrypted_path = directory / encrypted_name
    encrypted_path.write_bytes(encrypt_bytes(data))
    try:
        encrypted_path.chmod(0o600)
    except OSError:
        pass

    record: dict[str, Any] = {
        "asset_id": asset_id,
        "filename": safe_name,
        "mime_type": resolved_mime,
        "category": category,
        "size_bytes": len(data),
        "sha256_original": sha256_bytes(data),
        "storage_key": encrypted_name,
        "encryption": "Fernet (AES-128-CBC + HMAC-SHA256; authenticated encryption provided by Fernet)",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    record["ed25519_signature"] = sign_bytes(_canonical(record))
    manifest = directory / MANIFEST_NAME
    with manifest.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
    return record


def list_media(media_dir: str | Path = DEFAULT_MEDIA_DIR) -> list[dict[str, Any]]:
    manifest = Path(media_dir) / MANIFEST_NAME
    if not manifest.exists():
        return []
    records: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Media manifest is malformed at line {line_number}.") from exc
            records.append(record)
    return records


def verify_media(media_dir: str | Path = DEFAULT_MEDIA_DIR) -> list[dict[str, Any]]:
    directory = Path(media_dir)
    results = []
    for record in list_media(directory):
        signed_record = {k: v for k, v in record.items() if k != "ed25519_signature"}
        signature_ok = verify_signature(_canonical(signed_record), record.get("ed25519_signature", ""))
        file_path = directory / Path(str(record.get("storage_key", ""))).name
        file_ok = False
        reason = "Encrypted file missing"
        if file_path.exists():
            try:
                plaintext = decrypt_bytes(file_path.read_bytes())
                file_ok = sha256_bytes(plaintext) == record.get("sha256_original")
                reason = "Verified" if file_ok else "Original content hash mismatch"
            except Exception as exc:
                reason = f"Cannot decrypt/verify: {type(exc).__name__}"
        results.append({
            "asset_id": record.get("asset_id"),
            "filename": record.get("filename"),
            "mime_type": record.get("mime_type"),
            "size_bytes": record.get("size_bytes"),
            "signature_valid": signature_ok,
            "file_integrity_valid": file_ok,
            "status": "valid" if signature_ok and file_ok else "ALERT",
            "detail": reason,
        })
    return results


def load_media(asset_id: str, media_dir: str | Path = DEFAULT_MEDIA_DIR) -> tuple[dict[str, Any], bytes]:
    for record in list_media(media_dir):
        if record.get("asset_id") == asset_id:
            signed_record = {k: v for k, v in record.items() if k != "ed25519_signature"}
            if not verify_signature(_canonical(signed_record), record.get("ed25519_signature", "")):
                raise ValueError("Media metadata signature is invalid; refusing to load this asset.")
            path = Path(media_dir) / Path(str(record.get("storage_key", ""))).name
            data = decrypt_bytes(path.read_bytes())
            if sha256_bytes(data) != record.get("sha256_original"):
                raise ValueError("Media integrity check failed; refusing to use this asset.")
            return record, data
    raise FileNotFoundError("Media asset ID was not found in the manifest.")


def analyze_image_vlm(image_bytes: bytes, question: str,
                      model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct") -> str:
    """Run a local Hugging Face image-text-to-text model (downloads weights on first run)."""
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("VLM dependencies are missing. Install requirements-vlm.txt first.") from exc
    if not image_bytes:
        raise ValueError("No image bytes provided.")
    from PIL import Image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Cache the heavyweight pipeline in Streamlit's process when available.
    pipe = _get_vlm_pipeline(model_id)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    output = pipe(text=messages, images=[image], max_new_tokens=220, return_full_text=False)
    generated = output[0].get("generated_text", "")
    if isinstance(generated, list):
        generated = " ".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in generated)
    return str(generated).strip()


_VLM_CACHE: dict[str, Any] = {}
_WHISPER_CACHE: dict[str, Any] = {}
_DIFFUSION_CACHE: dict[str, Any] = {}

def _get_vlm_pipeline(model_id: str):
    if model_id not in _VLM_CACHE:
        _VLM_CACHE[model_id] = _load_vlm_pipeline(model_id)
    return _VLM_CACHE[model_id]


def _load_vlm_pipeline(model_id: str):
    from transformers import pipeline
    return pipeline("image-text-to-text", model=model_id, device=-1)



def analyze_audio_nvidia(audio_bytes: bytes, mime_type: str, question: str, api_key: str,
                         model: str = "google/gemma-3n-e4b-it", base_url: str = "https://integrate.api.nvidia.com/v1") -> str:
    """Call a NVIDIA-hosted audio-capable chat model using the documented audio_url payload.

    Availability depends on the chosen model and API endpoint supporting audio inputs.
    Audio is sent to the hosted provider; do not use for sensitive recordings without approval.
    """
    import base64
    from openai import OpenAI
    if not audio_bytes:
        raise ValueError("No audio bytes provided.")
    if len(audio_bytes) > 20 * 1024 * 1024:
        raise ValueError("Hosted audio analysis is limited to 20 MB per request in this demo.")
    if not api_key.strip():
        raise ValueError("Set NVIDIA_API_KEY before using hosted audio analysis.")
    mime = (mime_type or "audio/wav").split(";")[0].strip().lower()
    if mime not in {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/flac", "audio/ogg", "audio/mp4", "audio/m4a"}:
        mime = "audio/wav" if mime.startswith("audio/") else "audio/wav"
    data_url = f"data:{mime};base64,{base64.b64encode(audio_bytes).decode('ascii')}"
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
    response = client.chat.completions.create(
        model=model.strip(),
        messages=[{"role": "user", "content": [
            {"type": "text", "text": question.strip() or "Describe and summarize this audio."},
            {"type": "audio_url", "audio_url": {"url": data_url}},
        ]}],
        max_tokens=600, temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()

def transcribe_audio_alm(audio_bytes: bytes, filename: str = "audio.wav", model_size: str = "tiny") -> dict[str, Any]:
    """Transcribe speech locally with faster-whisper on CPU (model downloads on first run)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Audio model dependencies are missing. Install requirements-alm.txt first.") from exc
    import tempfile
    suffix = Path(filename).suffix or ".audio"
    if not audio_bytes:
        raise ValueError("No audio bytes provided.")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(audio_bytes)
            temp.flush()
            temp_path = temp.name
        model = _get_whisper_model(model_size)
        segments, info = model.transcribe(temp_path, beam_size=1, vad_filter=True)
        segment_list = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments]
    finally:
        if temp_path:
            try: Path(temp_path).unlink(missing_ok=True)
            except OSError: pass
    return {"language": getattr(info, "language", "unknown"),
            "language_probability": round(float(getattr(info, "language_probability", 0.0)), 3),
            "text": " ".join(s["text"] for s in segment_list).strip(), "segments": segment_list}


def _get_whisper_model(model_size: str):
    if model_size not in _WHISPER_CACHE:
        _WHISPER_CACHE[model_size] = _load_whisper_model(model_size)
    return _WHISPER_CACHE[model_size]


def _load_whisper_model(model_size: str):
    from faster_whisper import WhisperModel
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def generate_diffusion_image(prompt: str, negative_prompt: str = "", steps: int = 2,
                             model_id: str = "segmind/tiny-sd") -> bytes:
    """Generate a PNG with an optional local Diffusers model; CPU inference can be slow."""
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError as exc:
        raise RuntimeError("Diffusion dependencies are missing. Install requirements-diffusion.txt first.") from exc
    if not prompt.strip():
        raise ValueError("Enter an image-generation prompt.")
    pipe = _get_diffusion_pipeline(model_id)
    result = pipe(prompt=prompt.strip(), negative_prompt=negative_prompt.strip() or None,
                  num_inference_steps=max(1, min(int(steps), 4)), guidance_scale=1.0)
    image = result.images[0]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _get_diffusion_pipeline(model_id: str):
    if model_id not in _DIFFUSION_CACHE:
        _DIFFUSION_CACHE[model_id] = _load_diffusion_pipeline(model_id)
    return _DIFFUSION_CACHE[model_id]


def _load_diffusion_pipeline(model_id: str):
    import torch
    from diffusers import AutoPipelineForText2Image
    # CPU-compatible setting; generation is slow and memory-intensive on low-RAM machines.
    pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=torch.float32, use_safetensors=True)
    pipe = pipe.to("cpu")
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def extract_video_frames(video_bytes: bytes, filename: str = "video.mp4", max_frames: int = 3) -> list[tuple[float, bytes]]:
    """Extract up to three evenly-spaced JPEG frames. Requires requirements-video.txt and FFmpeg codec support."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video frame extraction dependency missing. Install requirements-video.txt first.") from exc
    import tempfile
    if not video_bytes:
        raise ValueError("No video bytes provided.")
    temp_path = None
    capture = None
    try:
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix or ".mp4", delete=False) as temp:
            temp.write(video_bytes)
            temp.flush()
            temp_path = temp.name
        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise RuntimeError("Could not open video. Check its codec; MP4/H.264 is the best-supported demo format.")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            raise RuntimeError("Video contains no readable frames.")
        count = max(1, min(int(max_frames), 3, frame_count))
        indices = sorted(set(round(i * (frame_count - 1) / max(count - 1, 1)) for i in range(count)))
        output = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok:
                success, encoded = cv2.imencode(".jpg", frame)
                if success:
                    output.append((round(index / fps, 2) if fps > 0 else float(index), encoded.tobytes()))
        return output
    finally:
        if capture is not None: capture.release()
        if temp_path:
            try: Path(temp_path).unlink(missing_ok=True)
            except OSError: pass
