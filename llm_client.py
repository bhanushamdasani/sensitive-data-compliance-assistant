"""
llm_client.py
-------------
Thin wrapper around the Gemini LLM used for the two
tasks that genuinely need language understanding:

  1. Generating a compliance/security narrative summary
  2. Answering free-form questions about the uploaded document
"""

import os
from typing import Dict, List, Optional, Any, Tuple
import google.generativeai as genai

MAX_DOC_CHARS_FOR_LLM = 12000  # keep prompts bounded / cost-controlled

SYSTEM_PROMPT = """You are a data-privacy and compliance assistant embedded in a
Sensitive Data Detection tool. You are given:
  - The text of an uploaded document (possibly truncated)
  - A structured list of sensitive-data findings already produced by a
    deterministic regex/checksum detector (type + count)
  - The overall risk classification already computed by a rule engine

Your job is to explain findings in plain language, discuss compliance
implications (e.g. India's DPDP Act, GDPR where relevant, PCI-DSS for
card data), and suggest concrete remediation steps. Do NOT claim to have
found sensitive data types that are not present in the provided findings
list. Be concise, structured, and professional."""


_SELECTED_MODEL = None

def _get_model_name() -> str:
    """
    Finds the highest-priority available Gemini model supported by your API key.
    """
    global _SELECTED_MODEL
    if _SELECTED_MODEL:
        return _SELECTED_MODEL
        
    default_model = "gemini-2.5-flash"
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return default_model
        
    try:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models()]
        short_names = [name.replace("models/", "") for name in models]
        
        priorities = [
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-flash-latest"
        ]
        
        for p in priorities:
            if p in short_names:
                _SELECTED_MODEL = p
                return _SELECTED_MODEL
                
        for name in short_names:
            if "flash" in name:
                _SELECTED_MODEL = name
                return _SELECTED_MODEL
    except Exception as e:
        print(f"[LLM Client] Error listing models: {e}")
        
    _SELECTED_MODEL = default_model
    return _SELECTED_MODEL

def _call_llm(system: str, user: str) -> Optional[str]:
    """
    Calls the configured Google Gemini model with system instructions and user prompt.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
        
    try:
        genai.configure(api_key=api_key)
        
        # Try the cached or default model name directly first to bypass listing overhead
        model_name = _SELECTED_MODEL or "gemini-2.5-flash"
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system
            )
            response = model.generate_content(user)
            return response.text
        except Exception as e:
            # Fallback to model listing if the default/cached model is unavailable
            print(f"[LLM Client] Direct call failed with {model_name}. Running listing fallback: {e}")
            model_name = _get_model_name()
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system
            )
            response = model.generate_content(user)
            return response.text
            
    except Exception as e: 
        return f"__ERROR__ LLM call failed: {e}"


def _findings_block(counts: Dict[str, int], risk: str, risk_reasons: List[str]) -> str:
    """
    Formats the PII detections and risk classification reasoning into a structured text segment.
    """
    lines = [f"Overall risk classification: {risk}"]
    lines.append("Reasons: " + " ".join(risk_reasons))
    lines.append("Detected entity counts:")
    if counts:
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {k}: {v}")
    else:
        lines.append("  (none)")
    return "\n".join(lines)


def generate_summary(
    document_text: str, counts: Dict[str, int], risk: str, risk_reasons: List[str]
) -> str:
    """
    Sends findings and document context to Gemini to write a natural language compliance brief.
    """
    findings_block = _findings_block(counts, risk, risk_reasons)
    truncated = document_text[:MAX_DOC_CHARS_FOR_LLM]

    user_prompt = f"""Here is the structured detection output:

{findings_block}

Here is the (possibly truncated) document text for context:
---
{truncated}
---

Write a compliance/security summary with these sections:
1. Compliance Observations
2. Security Risks
3. Suggested Remediation Steps

Keep it under ~300 words, use bullet points, be specific to what was found."""

    result = _call_llm(SYSTEM_PROMPT, user_prompt)
    if result is None:
        return _fallback_summary(counts, risk, risk_reasons)
    if result.startswith("__ERROR__"):
        return _fallback_summary(counts, risk, risk_reasons) + f"\n\n_(LLM unavailable: {result})_"
    return result


def answer_question(
    document_text: str,
    counts: Dict[str, int],
    risk: str,
    risk_reasons: List[str],
    question: str,
    chat_history: Optional[List[Dict]] = None,
    rag: Optional[Any] = None,
) -> Tuple[str, List[str]]:
    """
    Answers a compliance question using chat history and targeted RAG document references.
    """
    findings_block = _findings_block(counts, risk, risk_reasons)
    
    chunks = []
    if rag:
        try:
            chunks = rag.search(question, top_k=3)
            context_block = "\n---\n".join(chunks)
        except Exception as e:
            print(f"[LLM Client] RAG search failed: {e}")
            context_block = document_text[:MAX_DOC_CHARS_FOR_LLM]
            chunks = [context_block]
    else:
        context_block = document_text[:MAX_DOC_CHARS_FOR_LLM]
        chunks = [context_block]

    history_text = ""
    if chat_history:
        for turn in chat_history[-6:]:
            history_text += f"\n{turn['role'].capitalize()}: {turn['content']}"

    user_prompt = f"""Structured detection output:
{findings_block}

Relevant document context chunks:
---
{context_block}
---
{f"Recent conversation:{history_text}" if history_text else ""}

User question: {question}

Answer directly and concisely, grounded only in the document text and
the detection output above."""

    result = _call_llm(SYSTEM_PROMPT, user_prompt)
    if result is None:
        return _fallback_answer(counts, risk, risk_reasons, question), chunks
    if result.startswith("__ERROR__"):
        return _fallback_answer(counts, risk, risk_reasons, question) + f"\n\n_(LLM unavailable: {result})_", chunks
    return result, chunks


# ---------------------------------------------------------------------
# Deterministic fallbacks (no API key configured)
# ---------------------------------------------------------------------

def _fallback_summary(counts: Dict[str, int], risk: str, risk_reasons: List[str]) -> str:
    """
    Creates a basic template-based compliance summary when the Gemini API key is missing.
    """
    lines = [f"**Compliance Observations**"]
    if counts:
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"- Found {v} instance(s) of **{k}**.")
    else:
        lines.append("- No sensitive identifiers were detected in this document.")

    lines.append("\n**Security Risks**")
    lines.append(f"- Overall risk classification: **{risk}**")
    for r in risk_reasons:
        lines.append(f"- {r}")

    lines.append("\n**Suggested Remediation Steps**")
    if risk == "High Risk":
        lines.append("- Restrict document access to authorized personnel only.")
        lines.append("- Redact or tokenize identifiers (Aadhaar/PAN/card/bank/API keys) before sharing.")
        lines.append("- Rotate any exposed API keys/passwords immediately.")
        lines.append("- Review handling against DPDP Act / PCI-DSS / GDPR requirements as applicable.")
    elif risk == "Medium Risk":
        lines.append("- Limit distribution and apply access controls.")
        lines.append("- Mask email/phone/employee ID fields where not strictly required.")
    else:
        lines.append("- No immediate action required; maintain standard data-handling hygiene.")

    lines.append(
        "\n_Note: This is a template-based summary generated without an LLM API key configured. "
        "Set GEMINI_API_KEY for richer, natural-language analysis._"
    )
    return "\n".join(lines)


def _fallback_answer(counts: Dict[str, int], risk: str, risk_reasons: List[str], question: str) -> str:
    """
    Heuristically answers common questions (risk, summary, counts) when the Gemini API key is missing.
    """
    q = question.lower()
    if "how many" in q:
        for entity_type, count in counts.items():
            if entity_type.lower().split()[0] in q:
                return f"There are {count} instance(s) of {entity_type} in the document."
        return "Total sensitive items found: " + str(sum(counts.values())) + (
            "\n" + "\n".join(f"- {k}: {v}" for k, v in counts.items()) if counts else ""
        )
    if "risk" in q:
        return f"Overall risk classification: {risk}\n" + "\n".join(f"- {r}" for r in risk_reasons)
    if "summarize" in q or "summary" in q:
        return _fallback_summary(counts, risk, risk_reasons)
    if "what sensitive data" in q or "sensitive data exists" in q:
        if not counts:
            return "No sensitive data was detected in this document."
        return "Detected sensitive data types:\n" + "\n".join(f"- {k}: {v}" for k, v in counts.items())
    return (
        "I can answer this more precisely with an LLM API key configured "
        "(set GEMINI_API_KEY). In the meantime, here's what "
        "the detector found:\n" + "\n".join(f"- {k}: {v}" for k, v in counts.items())
    )