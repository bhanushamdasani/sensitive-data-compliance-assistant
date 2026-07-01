from typing import List, Dict, Tuple
from detectors import HIGH_RISK_TYPES, MEDIUM_RISK_TYPES

def classify_risk(
    findings: List[Dict],
    high_risk_set = None,
    medium_risk_set = None
) -> Tuple[str, Dict]:
    """
    Evaluates PII findings and determines the document risk classification (High, Medium, Low) and reasons.
    """
    hr = high_risk_set if high_risk_set is not None else HIGH_RISK_TYPES
    mr = medium_risk_set if medium_risk_set is not None else MEDIUM_RISK_TYPES

    high_count = sum(1 for f in findings if f["type"] in hr)
    medium_count = sum(1 for f in findings if f["type"] in mr)
    confidential_count = sum(1 for f in findings if f["type"] == "Confidential Business Language")

    high_types = {}
    medium_types = {}
    for f in findings:
        t = f["type"]
        if t in hr:
            high_types[t] = high_types.get(t, 0) + 1
        elif t in mr:
            medium_types[t] = medium_types.get(t, 0) + 1

    reasons = []

    if high_count > 0:
        risk = "High Risk"
        type_details = ", ".join(f"{k} ({v})" for k, v in sorted(high_types.items(), key=lambda kv: -kv[1]))
        reasons.append(f"{high_count} high-severity identifier(s) detected: {type_details}.")
    elif confidential_count >= 3:
        risk = "High Risk"
        reasons.append(f"Document repeatedly marks content as confidential ({confidential_count}x).")
    elif medium_count > 0 or confidential_count > 0:
        risk = "Medium Risk"
        if medium_count:
            type_details = ", ".join(f"{k} ({v})" for k, v in sorted(medium_types.items(), key=lambda kv: -kv[1]))
            reasons.append(f"{medium_count} medium-severity identifier(s) detected: {type_details}.")
        if confidential_count:
            reasons.append(f"{confidential_count} instance(s) of confidential-business language found.")
    else:
        risk = "Low Risk"
        reasons.append("No sensitive identifiers or confidentiality markers detected.")

    return risk, {
        "high_count": high_count,
        "medium_count": medium_count,
        "confidential_count": confidential_count,
        "reasons": reasons,
    }