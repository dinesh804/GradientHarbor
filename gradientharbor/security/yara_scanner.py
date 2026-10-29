"""Read-only YARA scanning for uploaded artifacts.

Files are read as bytes and never executed. Built-in rules are heuristic examples,
not a malware signature database. YARA matches require human interpretation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_RULE_BYTES = 256 * 1024

BUILTIN_RULES = r'''rule GH_Suspicious_PowerShell_Download_And_Execute
{
    meta:
        description = "Heuristic indicator for PowerShell download-and-execute patterns"
        category = "suspicious_script"
        confidence = "medium"
    strings:
        $ps1 = "DownloadString(" nocase ascii wide
        $ps2 = "DownloadFile(" nocase ascii wide
        $ps3 = "Invoke-Expression" nocase ascii wide
        $ps4 = "IEX(" nocase ascii wide
    condition:
        2 of them
}

rule GH_Suspicious_Credential_Access_Strings
{
    meta:
        description = "Multiple credential-access-related strings found"
        category = "credential_access_indicator"
        confidence = "low"
    strings:
        $s1 = "mimikatz" nocase ascii wide
        $s2 = "sekurlsa::logonpasswords" nocase ascii wide
        $s3 = "lsass" nocase ascii wide
        $s4 = "passwords.txt" nocase ascii wide
    condition:
        2 of them
}

rule GH_Suspicious_Encoded_PowerShell
{
    meta:
        description = "PowerShell encoded-command switch found"
        category = "suspicious_script"
        confidence = "low"
    strings:
        $s1 = "-EncodedCommand" nocase ascii wide
        $s2 = "-enc " nocase ascii wide
    condition:
        any of them
}

rule GH_Possible_Prompt_Injection_In_Artifact
{
    meta:
        description = "Instruction-override or prompt-extraction phrases found in artifact"
        category = "ai_prompt_injection_indicator"
        confidence = "low"
    strings:
        $s1 = "ignore all previous instructions" nocase ascii wide
        $s2 = "reveal the system prompt" nocase ascii wide
        $s3 = "disregard previous instructions" nocase ascii wide
        $s4 = "send secrets to" nocase ascii wide
    condition:
        any of them
}
'''


def _yara_module():
    try:
        import yara  # type: ignore
        return yara
    except ImportError as exc:
        raise RuntimeError(
            "YARA support is not installed. Install project dependencies with: pip install -r requirements.txt"
        ) from exc


def compile_rules(custom_rules: bytes | str | None = None):
    """Compile built-in rules and optionally a user-supplied YARA rule file."""
    yara = _yara_module()
    sources: dict[str, str] = {"gradientharbor_builtin": BUILTIN_RULES}
    if custom_rules:
        if isinstance(custom_rules, bytes):
            if len(custom_rules) > MAX_RULE_BYTES:
                raise ValueError(f"Rule file exceeds {MAX_RULE_BYTES // 1024} KiB limit.")
            try:
                custom_text = custom_rules.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError("YARA rule files must be UTF-8 text.") from exc
        else:
            custom_text = custom_rules
            if len(custom_text.encode("utf-8")) > MAX_RULE_BYTES:
                raise ValueError(f"Rule file exceeds {MAX_RULE_BYTES // 1024} KiB limit.")
        sources["user_supplied_rules"] = custom_text
    try:
        return yara.compile(sources=sources)
    except Exception as exc:
        raise ValueError(f"YARA rule compilation failed: {type(exc).__name__}: {exc}") from exc


def scan_bytes(data: bytes, filename: str = "uploaded-artifact", custom_rules: bytes | str | None = None) -> dict[str, Any]:
    """Scan artifact bytes without writing or executing them."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"Artifact exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MiB scan limit.")
    rules = compile_rules(custom_rules)
    try:
        matches = rules.match(data=bytes(data), timeout=5)
    except Exception as exc:
        raise RuntimeError(f"YARA scan failed: {type(exc).__name__}: {exc}") from exc

    findings = []
    for match in matches:
        meta = dict(getattr(match, "meta", {}) or {})
        findings.append({
            "rule": str(match.rule),
            "namespace": str(getattr(match, "namespace", "")),
            "description": str(meta.get("description", "No rule description supplied.")),
            "category": str(meta.get("category", "custom_rule_match")),
            "confidence": str(meta.get("confidence", "unspecified")),
            "tags": list(getattr(match, "tags", []) or []),
        })
    return {
        "filename": Path(filename).name[:255],
        "file_size_bytes": len(data),
        "sha256": __import__("hashlib").sha256(data).hexdigest(),
        "engine": f"YARA {getattr(_yara_module(), '__version__', 'version unavailable')}",
        "ruleset": "built-in heuristic rules" + (" + custom rules" if custom_rules else ""),
        "match_count": len(findings),
        "status": "matches_found" if findings else "no_matches",
        "findings": findings,
        "limitations": "A match is an indicator, not proof of malware or compromise. No match does not prove an artifact is safe. Only the supplied rules were evaluated.",
    }
