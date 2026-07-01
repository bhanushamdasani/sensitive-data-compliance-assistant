import re
from typing import List, Dict

# Risk classifications for the Risk Engine
HIGH_RISK_TYPES = {
    "PAN Number", "Aadhaar Number", "Credit Card Number", 
    "Bank Account Number", "API Key / Secret", "Password", "IFSC Code"
}
MEDIUM_RISK_TYPES = {
    "Email Address", "Phone Number", "Employee ID"
}

PATTERNS = {
    "PAN Number": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "Email Address": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "IFSC Code": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "Employee ID": re.compile(r"\b(?:EMP|EMPID|EID)[-_]?\d{3,8}\b", re.IGNORECASE),
    "Phone Number": re.compile(r"\b(?:\+?91[\s-]?)?[6789]\d{9}\b"),
    "API Key / Secret": re.compile(r"\b(?:ghp_[a-zA-Z0-9]{36}|sk-[a-zA-Z0-9\-_]{20,}|AKIA[0-9A-Z]{16}|sk_(?:live|test)_[a-zA-Z0-9_]{20,})\b"),
    "Password": re.compile(r"(?i)\bpassword\b\s*[:=]\s*['\"]?\S{4,}['\"]?"),
}

CONFIDENTIAL_KEYWORDS = [
    "confidential", "internal use only", "do not distribute", "trade secret",
    "proprietary", "strictly private", "not for public release",
    "restricted access", "for internal circulation only", "commercially sensitive",
]

DIGIT_RUN = re.compile(r"\b(?:\d[ \-]?){9,19}\b")

# Verhoeff tables for Aadhaar checksum validation
D_TABLE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

P_TABLE = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
]

def _verhoeff_check(digits: str) -> bool:
    """
    Verifies if a numeric digit sequence has a valid Verhoeff checksum (used to validate Aadhaar numbers).
    """
    try:
        if not digits.isdigit():
            return False
        c = 0
        for i, item in enumerate(reversed(digits)):
            c = D_TABLE[c][P_TABLE[i % 8][int(item)]]
        return c == 0
    except Exception:
        return False

def _luhn_check(digits: str) -> bool:
    """
    Verifies if a numeric digit sequence has a valid Luhn checksum (used to validate credit card numbers).
    """
    total = 0
    reverse_digits = digits[::-1]
    for i, ch in enumerate(reverse_digits):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9: n -= 9
        total += n
    return total % 10 == 0

def find_numeric_entities(text: str) -> List[Dict]:
    """
    Scans document text for numbers and verifies them as Aadhaar, Credit Cards, Bank Accounts, or Phones.
    """
    findings = []
    for m in DIGIT_RUN.finditer(text):
        raw = m.group().strip()
        digits = re.sub(r"[ \-]", "", raw)
        length = len(digits)

        window_start = max(0, m.start() - 25)
        context = text[window_start:m.start()].lower()
        has_account_context = "a/c" in context or "account" in context or "acct" in context

        entity_type = None
        if 9 <= length <= 18 and has_account_context:
            entity_type = "Bank Account Number"
        elif 13 <= length <= 19 and _luhn_check(digits):
            entity_type = "Credit Card Number"
        elif length == 12 and _verhoeff_check(digits):
            entity_type = "Aadhaar Number"
        elif length == 10 and digits[0] in "6789":
            entity_type = "Phone Number"

        if entity_type:
            findings.append({"type": entity_type, "value": raw, "start": m.start(), "end": m.end()})
    return findings

def find_pattern_entities(text: str) -> List[Dict]:
    """
    Scans document text for regex patterns like PAN cards, emails, passwords, and API secrets.
    """
    findings = []
    for label, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            findings.append({"type": label, "value": m.group().strip(), "start": m.start(), "end": m.end()})
    return findings

def find_confidential_language(text: str) -> List[Dict]:
    """
    Scans document text for keywords indicating proprietary, private, or restricted business language.
    """
    findings = []
    for kw in CONFIDENTIAL_KEYWORDS:
        for m in re.finditer(re.escape(kw), text.lower()):
            findings.append({"type": "Confidential Business Language", "value": text[m.start():m.end()], "start": m.start(), "end": m.end()})
    return findings

def detect_sensitive_data(text: str) -> List[Dict]:
    """
    Runs all numeric, regex, and confidentiality pattern checks, removes duplicates, and sorts findings.
    """
    findings = find_numeric_entities(text) + find_pattern_entities(text) + find_confidential_language(text)
    
    seen = set()
    deduped = []
    for f in findings:
        key = (f["type"], f["start"], f["end"])
        if key not in seen:
            seen.add(key)
            if f["type"] in HIGH_RISK_TYPES:
                f["risk"] = "high"
            elif f["type"] in MEDIUM_RISK_TYPES:
                f["risk"] = "medium"
            else:
                f["risk"] = "low"
            deduped.append(f)

    deduped.sort(key=lambda f: f["start"])
    return deduped

def summarize_findings(findings: List[Dict]) -> Dict[str, int]:
    """
    Tallies the total occurrences of each detected sensitive entity type.
    """
    counts = {}
    for f in findings:
        counts[f["type"]] = counts.get(f["type"], 0) + 1
    return counts