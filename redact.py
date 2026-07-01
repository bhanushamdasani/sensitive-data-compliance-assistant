import io
import pandas as pd
from typing import List, Dict
from detectors import detect_sensitive_data

def mask_value(entity_type: str, value: str) -> str:
    """
    Creates a masked copy of a sensitive string, keeping the last 4 digits if possible.
    """
    digits_only = "".join(ch for ch in value if ch.isdigit())
    if digits_only and len(digits_only) >= 4:
        last4 = digits_only[-4:]
        return f"[{entity_type.upper()} REDACTED: ***{last4}]"
    return f"[{entity_type.upper()} REDACTED]"

def redact_text(text: str, findings: List[Dict]) -> str:
    """
    Replaces detected PII strings in text with their masked equivalents.
    """
    redacted = text
    for f in sorted(findings, key=lambda f: f["start"], reverse=True):
        mask = mask_value(f["type"], f["value"])
        redacted = redacted[: f["start"]] + mask + redacted[f["end"] :]
    return redacted

def _redact_cell_value(val: str) -> str:
    """
    Scans a single table cell text value and redacts any detected PII.
    """
    if not val or val == "nan":
        return val
    findings = detect_sensitive_data(val)
    if findings:
        return redact_text(val, findings)
    return val

def redact_csv(file_bytes: bytes) -> bytes:
    """
    Parses CSV bytes, runs cell-level PII scans, and returns the redacted CSV file bytes.
    """
    try:
        # Load CSV into DataFrame
        df = pd.read_csv(io.BytesIO(file_bytes))
        
        # Apply redaction to each cell
        for col in df.columns:
            df[col] = df[col].astype(str).apply(_redact_cell_value)
            
        # Convert back to CSV bytes
        output = io.BytesIO()
        df.to_csv(output, index=False)
        return output.getvalue()
    except Exception as e:
        print(f"[Redact] Failed to redact CSV: {e}")
        return file_bytes