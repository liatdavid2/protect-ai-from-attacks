import re
from typing import Dict, Any

LEAK_PATTERNS = [
    re.compile(r"\bsystem prompt\b", re.IGNORECASE),
    re.compile(r"\bdeveloper instructions?\b", re.IGNORECASE),
    re.compile(r"\bhidden instructions?\b", re.IGNORECASE),
    re.compile(r"\binternal instructions?\b", re.IGNORECASE),
    re.compile(r"\bhidden policies?\b", re.IGNORECASE),
    re.compile(r"\bconfidential instructions?\b", re.IGNORECASE),
    re.compile(r"\bsecret internal behavior\b", re.IGNORECASE),
    re.compile(r"\byou are .* assistant\b", re.IGNORECASE),
    re.compile(r"\byour role is\b", re.IGNORECASE),
    re.compile(r"\byou must always\b", re.IGNORECASE),
    re.compile(r"\bdo not disclose\b", re.IGNORECASE),
]


def detect_system_prompt_leakage_regex(text: str) -> bool:
    return any(pattern.search(text) for pattern in LEAK_PATTERNS)


def run_system_prompt_leakage_guard(text: str, model_proba: float, threshold: float = 0.7) -> Dict[str, Any]:
    if detect_system_prompt_leakage_regex(text):
        return {
            "action": "block",
            "reason": "Potential system prompt leakage detected by regex",
            "response": None,
        }

    if model_proba >= threshold:
        return {
            "action": "block",
            "reason": "Potential system prompt leakage detected by model",
            "response": None,
        }

    return {
        "action": "allow",
        "reason": None,
        "response": text,
    }
