"""
Basic sanity tests. Run with: pytest test_detectors.py -v
"""

from detectors import detect_sensitive_data, summarize_findings
from risk_engine import classify_risk


SAMPLE = """
Email: john.doe@example.com
Phone: 9876543210
PAN Number: ABCDE1234F
Aadhaar Number: 2345 6789 1008
api_key: sk_test_51H8xYzT3example0000000000
This document is CONFIDENTIAL and strictly private.
"""


def test_email_detected():
    findings = detect_sensitive_data(SAMPLE)
    types = {f["type"] for f in findings}
    assert "Email Address" in types


def test_pan_detected():
    findings = detect_sensitive_data(SAMPLE)
    types = {f["type"] for f in findings}
    assert "PAN Number" in types


def test_aadhaar_detected():
    findings = detect_sensitive_data(SAMPLE)
    types = {f["type"] for f in findings}
    assert "Aadhaar Number" in types


def test_api_key_detected():
    findings = detect_sensitive_data(SAMPLE)
    types = {f["type"] for f in findings}
    assert "API Key / Secret" in types


def test_risk_classification_high():
    findings = detect_sensitive_data(SAMPLE)
    risk, meta = classify_risk(findings)
    assert risk == "High Risk"


def test_no_sensitive_data_is_low_risk():
    findings = detect_sensitive_data("This is a plain, harmless memo about lunch plans.")
    risk, meta = classify_risk(findings)
    assert risk == "Low Risk"


def test_credit_card_luhn_validated():
    text = "Test card: 4111 1111 1111 1111"
    findings = detect_sensitive_data(text)
    types = {f["type"] for f in findings}
    assert "Credit Card Number" in types


def test_invalid_card_number_not_flagged_as_card():
    # Fails Luhn check -> should not be classified as a credit card
    text = "Reference number: 1234 5678 9012 3456"
    findings = detect_sensitive_data(text)
    card_findings = [f for f in findings if f["type"] == "Credit Card Number"]
    assert len(card_findings) == 0


def test_invalid_aadhaar_number_not_flagged():
    # Fails Verhoeff check -> should not be classified as Aadhaar
    text = "Invalid Aadhaar: 2345 6789 0123"
    findings = detect_sensitive_data(text)
    aadhaar_findings = [f for f in findings if f["type"] == "Aadhaar Number"]
    assert len(aadhaar_findings) == 0

