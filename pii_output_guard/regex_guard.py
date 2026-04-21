import re
from typing import Dict, Any

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3}[-.\s]?\d{4,}\b")
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

AWS_KEY_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
HF_TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")
GITHUB_TOKEN_RE = re.compile(r"\bghp_[A-Za-z0-9]{20,}\b")
OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b", re.IGNORECASE)
BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._\-]+\b", re.IGNORECASE)
SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(api[_\- ]?key|secret|token|password|connection[_\- ]?string)\s*[:=]\s*\S+",
    re.IGNORECASE,
)

LEAK_HINTS = [
    "system prompt",
    "hidden prompt",
    "hidden instructions",
    "internal instructions",
    "developer instructions",
    "secret key",
    "access token",
    "password",
]


def redact_pii(text: str) -> str:
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = CREDIT_CARD_RE.sub("[REDACTED_CARD]", text)
    text = SSN_RE.sub("[REDACTED_ID]", text)
    return text


def detect_secret_regex(text: str) -> bool:
    if AWS_KEY_RE.search(text):
        return True
    if HF_TOKEN_RE.search(text):
        return True
    if GITHUB_TOKEN_RE.search(text):
        return True
    if OPENAI_KEY_RE.search(text):
        return True
    if BEARER_RE.search(text):
        return True
    if SECRET_ASSIGNMENT_RE.search(text):
        return True

    lower_text = text.lower()
    for hint in LEAK_HINTS:
        if hint in lower_text:
            return True
    return False


def run_regex_guard(text: str) -> Dict[str, Any]:
    if detect_secret_regex(text):
        return {
            "action": "block",
            "reason": "Potential secret leakage detected by regex",
            "response": None,
        }

    redacted_text = redact_pii(text)
    if redacted_text != text:
        return {
            "action": "redact",
            "reason": "PII detected and redacted by regex",
            "response": redacted_text,
        }

    return {
        "action": "allow",
        "reason": None,
        "response": text,
    }
