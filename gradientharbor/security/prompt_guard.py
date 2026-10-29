"""Defense-in-depth input and output guardrails for GradientHarbor.

Deterministic heuristics are explainable but not complete jailbreak detection.
All user prompts, retrieved passages, and model outputs remain untrusted.
"""
from __future__ import annotations
import base64
import re
import unicodedata
from typing import Any

MAX_REQUEST_CHARS = 12_000
MAX_UNTRUSTED_CHARS = 30_000

_PATTERNS: list[tuple[str, int, str]] = [
    ("instruction_override", 5, r"\b(ignore|disregard|forget|override|bypass)\b.{0,90}\b(previous|prior|above|all|system|developer|safety|security)\b.{0,60}\b(instruction|prompt|rule|policy|message)s?\b"),
    ("system_prompt_extraction", 5, r"\b(reveal|print|show|dump|leak|expose|repeat|quote)\b.{0,60}\b(system prompt|hidden prompt|developer message|secret instructions|internal rules)\b"),
    ("jailbreak_persona", 5, r"\b(DAN mode|developer mode|unrestricted mode|no restrictions|act as an unfiltered|disable safety|uncensored mode|jailbreak mode)\b"),
    ("credential_exfiltration", 5, r"\b(exfiltrate|steal|leak|send me|dump|print|reveal)\b.{0,80}\b(api key|password|credential|secret key|access token|private key)\b"),
    ("tool_policy_bypass", 5, r"\b(bypass|disable|circumvent|ignore)\b.{0,60}\b(sql guard|validator|firewall|authorization|access control|tool restrictions|permission checks)\b"),
    ("role_spoofing", 3, r"(?m)^\s*(<\|?(system|developer|assistant)\|?>|system\s*:|developer\s*:|assistant\s+to=)"),
    ("indirect_instruction", 3, r"\b(when an ai reads this|instructions for the assistant|ignore the user's request|do not tell the user|secretly execute)\b"),
    ("unsafe_external_action", 4, r"\b(delete|wipe|exfiltrate|upload|email|transfer)\b.{0,80}\b(all records|customer data|database|credentials|private files|secret data)\b"),
    ("policy_evasion", 3, r"\b(do not apply|bypass|evade|disable)\b.{0,60}\b(policy|guardrail|safety filter|moderation|verification)\b"),
    ("prompt_boundary_attack", 3, r"(\[/?INST\]|<\|im_(start|end)\|>|<\|fim_(prefix|middle|suffix)\|>|###\s*(system|developer)\s*instructions)"),
]

# Common credential formats. This is a leak detector, not a general PII detector.
_SECRET_PATTERNS: list[tuple[str, str]] = [
    ("nvidia_api_key_like", r"\bnvapi-[A-Za-z0-9_-]{20,}\b"),
    ("openai_api_key_like", r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    ("aws_access_key_like", r"\bAKIA[0-9A-Z]{16}\b"),
    ("private_key_block", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ("generic_secret_assignment", r"(?i)\b(?:api[_ -]?key|password|secret[_ -]?key|access[_ -]?token)\s*[:=]\s*['\"]?[^\s'\"]{12,}"),
]

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    # Remove zero-width and other formatting controls commonly used to evade simple filters.
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf" and ch not in "\x00")

def _signals(text: str) -> list[tuple[str, int]]:
    clean = _normalize(text)[:MAX_UNTRUSTED_CHARS]
    found: list[tuple[str, int]] = []
    for name, weight, pattern in _PATTERNS:
        if re.search(pattern, clean, flags=re.IGNORECASE | re.DOTALL):
            found.append((name, weight))
    # If a long base64-like token is present, inspect a small decoded UTF-8 payload for
    # recognizable attacks. Do not execute or evaluate decoded content.
    for token in re.findall(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{80,4096}={0,2}(?![A-Za-z0-9+/])", clean)[:3]:
        try:
            decoded = base64.b64decode(token, validate=True).decode("utf-8", errors="ignore")[:4000]
        except Exception:
            continue
        for name, weight, pattern in _PATTERNS:
            if re.search(pattern, _normalize(decoded), flags=re.IGNORECASE | re.DOTALL):
                found.append(("encoded_" + name, min(weight, 4)))
    # Preserve order while deduplicating.
    return list(dict.fromkeys(found))

def scan_untrusted_text(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        text = str(text)
    signals = _signals(text)
    score = min(10, sum(weight for _, weight in signals))
    names = [name for name, _ in signals]
    return {
        "suspicious": bool(names),
        "signals": names,
        "risk_score": score,
        "severity": "high" if score >= 5 else ("medium" if score >= 3 else "low"),
        "action": "block" if score >= 5 else ("review" if score >= 3 else "allow_with_caution"),
        "note": "Heuristic screening only; no rule match does not prove safety. Keep untrusted content separate from system instructions.",
    }

def screen_user_request(text: str) -> dict[str, Any]:
    clean = _normalize(text)
    result = scan_untrusted_text(clean)
    too_long = len(clean) > MAX_REQUEST_CHARS
    result["blocked"] = bool(too_long or result["risk_score"] >= 5)
    result["requires_review"] = bool(not result["blocked"] and result["risk_score"] >= 3)
    if too_long:
        result["signals"] = list(result["signals"]) + ["request_length_limit"]
        result["risk_score"] = 10
        result["severity"] = "high"
        result["action"] = "block"
    result["normalized_length"] = len(clean)
    return result

def screen_model_output(text: str) -> dict[str, Any]:
    """Detect credential-shaped strings before rendering model output to the user."""
    clean = text or ""
    matches = [name for name, pattern in _SECRET_PATTERNS if re.search(pattern, clean)]
    return {
        "blocked": bool(matches),
        "signals": matches,
        "action": "block_output_for_review" if matches else "no_secret_pattern_match",
        "note": "Pattern-based output screening is not a guarantee against secret disclosure.",
    }
