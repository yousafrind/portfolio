import re
from typing import Tuple, Dict, List

# Simple regex-based PII redaction. This is best-effort and not a
# substitute for a full PII detection pipeline. Patterns can be extended.

PATTERNS = {
    "email": (re.compile(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,6}"), "[REDACTED_EMAIL]"),
    # international-ish phone numbers, simple match for common formats
    "phone": (re.compile(r"\+?\d[\d\-\.\s()]{7,}\d"), "[REDACTED_PHONE]"),
    # US SSN-like
    "ssn": (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    # credit card-ish numbers (13-16 digits, allowing spaces/dashes)
    "credit_card": (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[REDACTED_CREDIT_CARD]"),
    # dates common formats MM/DD/YYYY or DD-MM-YYYY, YYYY-MM-DD etc.
    "date": (re.compile(r"\b(?:\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4})\b"), "[REDACTED_DATE]"),
    # simple MRN / patient id tokens
    "mrn": (re.compile(r"\b(MRN|Patient\s*ID|PID)[:#\s]*[A-Za-z0-9\-]+\b", re.I), "[REDACTED_MRN]"),
}

# Field-style name capture: "Name: John Doe" -> redact value
NAME_FIELD_RE = re.compile(r"\b(?:Patient\s*Name|Name|Full\s*Name)\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", re.I)


def redact_for_api(text: str, types: List[str] = None) -> Tuple[str, Dict[str, int]]:
    """Redact PII from text using configured patterns.

    Returns (redacted_text, counts) where counts maps pattern name -> replacements made.
    """
    if not text:
        return text, {}
    counts: Dict[str, int] = {}
    out = text

    sel = types if types else list(PATTERNS.keys())
    for key in sel:
        pat, token = PATTERNS[key]
        new, n = pat.subn(token, out)
        if n:
            counts[key] = counts.get(key, 0) + n
            out = new

    # redact name fields like "Name: John Doe"
    def _name_repl(m: re.Match) -> str:
        name_val = m.group(1)
        return m.group(0).replace(name_val, "[REDACTED_NAME]")

    out, n = NAME_FIELD_RE.subn(_name_repl, out)
    if n:
        counts["name_field"] = counts.get("name_field", 0) + n

    return out, counts


def redact_query(query: str) -> str:
    """Convenience wrapper to redact queries sent to external search/APIs.

    This returns a redacted query string. It uses the default PATTERNS and
    NAME_FIELD_RE.
    """
    redacted, _ = redact_for_api(query)
    return redacted


def redact_prompt(prompt: str) -> str:
    """Redact a prompt before sending to external LLMs.

    Same behavior as redact_for_api; keeps return signature simple.
    """
    redacted, _ = redact_for_api(prompt)
    return redacted
