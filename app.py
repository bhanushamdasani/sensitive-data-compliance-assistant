import streamlit as st
import os
import pandas as pd
import altair as alt
from dotenv import load_dotenv
from file_parser import extract_text
from detectors import detect_sensitive_data, summarize_findings
from risk_engine import classify_risk
from redact import redact_text, redact_csv, mask_value
from llm_client import generate_summary, answer_question
from rag_engine import DocumentRAG
import audit_log

# 1. Load environment variables
load_dotenv()

# 2. Page Config
st.set_page_config(page_title="AI Compliance Checker", layout="wide", page_icon="🛡️")

# 3. Modern UI Styling (Glassmorphism, Hover Slides, Custom Button styling)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #f8fafc;
    }
    
    .header-gradient {
        background: linear-gradient(90deg, #0f172a, #1e293b, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .subheader-text {
        color: #475569;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .risk-card {
        padding: 20px;
        border-radius: 16px;
        color: white;
        font-weight: 600;
        margin-bottom: 0px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #ef4444, #b91c1c);
        border: 1px solid #f87171;
    }
    .risk-medium {
        background: linear-gradient(135deg, #f97316, #c2410c);
        border: 1px solid #fb923c;
    }
    .risk-low {
        background: linear-gradient(135deg, #22c55e, #15803d);
        border: 1px solid #4ade80;
    }

    /* --- Custom Premium Sidebar Buttons --- */
    div[data-testid="stSidebar"] button {
        border-radius: 12px !important;
        padding: 14px 20px !important;
        font-weight: 600 !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-align: left !important;
        justify-content: flex-start !important;
        width: 100% !important;
        border: 1px solid transparent !important;
        margin-bottom: 8px !important;
    }

    /* Secondary (Inactive) Button Style */
    div[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: #f1f5f9 !important;
        color: #475569 !important;
        border: 1px solid #e2e8f0 !important;
    }
    div[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #e2e8f0 !important;
        color: #0f172a !important;
        transform: translateX(6px) !important;
        border-color: #cbd5e1 !important;
    }

    /* Primary (Active) Button Style */
    div[data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
    }
    div[data-testid="stSidebar"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
        transform: translateX(6px) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3) !important;
    }

    /* Custom styles for Top Columns button */
    div[data-testid="column"] button {
        border-radius: 16px !important;
        padding: 18px 24px !important;
        font-weight: 600 !important;
        height: 60px !important;
    }

    /* --- Chatbot styling for vertical alignment and speech bubble formatting --- */
    div[data-testid="stChatMessage"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 18px 20px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        display: flex !important;
        align-items: flex-start !important;
        gap: 16px !important;
    }
    
    /* Align custom icons vertically centered with the message text */
    div[data-testid="stChatMessage"] div[data-testid="stChatMessageAvatar"] {
        margin-top: 0px !important;
        align-self: flex-start !important;
    }
    
    div[data-testid="stChatMessage"] div[class*="stMarkdown"] {
        align-self: center !important;
        width: 100% !important;
    }

    /* Sidebar container alignment spacing */
    div[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# 4. Initialize State Defaults in Session State
if "high_risk_types" not in st.session_state:
    from detectors import HIGH_RISK_TYPES, MEDIUM_RISK_TYPES
    st.session_state.high_risk_types = set(HIGH_RISK_TYPES)
    st.session_state.medium_risk_types = set(MEDIUM_RISK_TYPES)

if "registry" not in st.session_state:
    st.session_state.registry = {}

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Insights & Summary"

# 5. Sidebar Layout & Action Navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=64)
    st.markdown("""
        <h3 style="font-weight: 700; color: #0f172a; margin-top: 12px; margin-bottom: 0px; font-size: 1.3rem;">AI Compliance Checker</h3>
        <p style="font-size: 0.85rem; color: #64748b; margin-top: 4px; margin-bottom: 24px; line-height: 1.35;">
            Scan sensitive identifiers, assess compliance risks, and interact with documents securely.
        </p>
    """, unsafe_allow_html=True)
    
    active = st.session_state.active_tab
    if active not in ["📊 Insights & Summary", "💬 Chatbot", "📜 Audit History"]:
        st.session_state.active_tab = "📊 Insights & Summary"
        active = "📊 Insights & Summary"

    btn_insights = st.button(
        "📊 Insights & Summary",
        key="nav_insights",
        type="primary" if active == "📊 Insights & Summary" else "secondary"
    )
    btn_chatbot = st.button(
        "💬 Chatbot",
        key="nav_chatbot",
        type="primary" if active == "💬 Chatbot" else "secondary"
    )
    btn_audit = st.button(
        "📜 Audit History",
        key="nav_audit",
        type="primary" if active == "📜 Audit History" else "secondary"
    )
    
    if btn_insights and active != "📊 Insights & Summary":
        st.session_state.active_tab = "📊 Insights & Summary"
        st.rerun()
    elif btn_chatbot and active != "💬 Chatbot":
        st.session_state.active_tab = "💬 Chatbot"
        st.rerun()
    elif btn_audit and active != "📜 Audit History":
        st.session_state.active_tab = "📜 Audit History"
        st.rerun()

# Set current active navigation value
nav = st.session_state.active_tab

# --- Main App Header ---
st.markdown('<h1 class="header-gradient">Sensitive Data Detection & Compliance Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Secure PII/PCI-DSS scanner with custom rules and in-memory RAG chatbot.</p>', unsafe_allow_html=True)

# --- Top Section: File Upload & File Selector ---
st.markdown("### 📁 Scanned Documents Registry")
uploaded_files = st.file_uploader(
    "Upload files to scan", 
    type=["pdf", "txt", "csv", "png", "jpg", "jpeg"], 
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# Process uploads
if uploaded_files:
    current_names = {f.name for f in uploaded_files}
    for name in list(st.session_state.registry.keys()):
        if name not in current_names:
            del st.session_state.registry[name]
            
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        if filename not in st.session_state.registry:
            with st.spinner(f"Scanning and indexing {filename}..."):
                try:
                    file_bytes = uploaded_file.read()
                    text = extract_text(filename, file_bytes)
                    findings = detect_sensitive_data(text)
                    counts = summarize_findings(findings)
                    
                    # Run classifier
                    risk, meta = classify_risk(
                        findings, 
                        st.session_state.high_risk_types, 
                        st.session_state.medium_risk_types
                    )
                    
                    # Initialize RAG Engine
                    rag = DocumentRAG(text)
                    api_key = os.environ.get("GEMINI_API_KEY")
                    rag.initialize(api_key=api_key)
                    
                    st.session_state.registry[filename] = {
                        "filename": filename,
                        "file_bytes": file_bytes,
                        "text": text,
                        "findings": findings,
                        "counts": counts,
                        "risk": risk,
                        "meta": meta,
                        "rag": rag,
                        "summary": None,
                        "chat_history": []
                    }
                    
                    # Log event
                    audit_log.log_event("document_scanned", filename, {
                        "risk": risk,
                        "counts": counts
                    })
                except Exception as e:
                    st.error(f"Failed to scan {filename}: {e}")

# Selector dropdown (rendered if documents are active)
active_file = None
if st.session_state.registry:
    st.markdown("---")
    col_sel, _ = st.columns([1, 2])
    with col_sel:
        active_file = st.selectbox(
            "Select an uploaded file to inspect:",
            options=list(st.session_state.registry.keys())
        )

def compile_chat_history(chat_history) -> str:
    """
    Converts the in-memory chat conversation list into a single human-readable text string for export.
    """
    lines = []
    for msg in chat_history:
        role = "Human" if msg["role"] == "user" else "AI"
        lines.append(f"{role}: {msg['content']}")
        if "sources" in msg and msg["sources"]:
            lines.append("References / Source chunks used:")
            for idx, src in enumerate(msg["sources"]):
                lines.append(f"  [{idx+1}] {src.strip()}")
        lines.append("\n" + "="*50 + "\n")
    return "\n".join(lines)

# --- Render Tab View Based on Sidebar Selection ---
if nav == "📊 Insights & Summary":
    if st.session_state.registry:
        if active_file:
            doc = st.session_state.registry[active_file]
            
            # Top Actions Row: Risk Card & Secure Exports side-by-side
            col_risk, col_export = st.columns([3, 1])
            
            risk_class = doc["risk"]
            color_class = "risk-high" if risk_class == "High Risk" else "risk-medium" if risk_class == "Medium Risk" else "risk-low"
            emoji = "🔴" if risk_class == "High Risk" else "🟠" if risk_class == "Medium Risk" else "🟢"
            
            with col_risk:
                st.markdown(f"""
                    <div class="risk-card {color_class}">
                        <span style="font-size: 1.25rem;">{emoji} Risk Assessment: <strong>{risk_class}</strong></span>
                        <span style="background: rgba(255,255,255,0.25); padding: 4px 12px; border-radius: 20px; font-size: 0.9rem;">DLP Rule Engine Active</span>
                    </div>
                """, unsafe_allow_html=True)
                
            with col_export:
                lower_name = doc["filename"].lower()
                if lower_name.endswith(".csv"):
                    with st.spinner("Redacting CSV..."):
                        redacted_csv_bytes = redact_csv(doc["file_bytes"])
                    st.download_button(
                        label="⬇️ Download Redacted CSV",
                        data=redacted_csv_bytes,
                        file_name=f"redacted_{doc['filename']}",
                        mime="text/csv",
                        use_container_width=True,
                        on_click=lambda: audit_log.log_event("redaction_downloaded", doc["filename"], {"format": "CSV"})
                    )
                else:
                    redacted_text_str = redact_text(doc["text"], doc["findings"])
                    download_name = f"redacted_{doc['filename']}"
                    if lower_name.endswith(".pdf"):
                        download_name = download_name.replace(".pdf", ".txt")
                    elif not download_name.endswith(".txt"):
                        download_name += ".txt"
                        
                    st.download_button(
                        label="⬇️ Download Redacted TXT",
                        data=redacted_text_str,
                        file_name=download_name,
                        mime="text/plain",
                        use_container_width=True,
                        on_click=lambda: audit_log.log_event("redaction_downloaded", doc["filename"], {"format": "TXT"})
                    )
            
            # Risk Reasoning
            if doc["meta"]["reasons"]:
                st.write("**Reasoning for risk classification:**")
                for reason in doc["meta"]["reasons"]:
                    st.markdown(f"- {reason}")
            
            # AI Narrative Compliance Summary Section
            st.markdown("---")
            st.markdown("#### 🤖 AI Generated Compliance Brief")
            st.write("Provides a detailed narrative covering compliance implications (GDPR, India's DPDP Act, PCI-DSS) and specific suggested remediation steps.")
            
            if not doc["summary"]:
                with st.spinner("Generating brief with Google Gemini..."):
                    brief = generate_summary(
                        doc["text"], 
                        doc["counts"], 
                        doc["risk"], 
                        doc["meta"]["reasons"]
                    )
                    doc["summary"] = brief
                    audit_log.log_event("summary_generated", doc["filename"], {})
                    
            st.markdown(doc["summary"])
            
            # Findings Metrics, Table & Chart Side-by-Side (Rule Checker Dashboard)
            st.markdown("---")
            if doc["counts"]:
                st.markdown("##### Detected Findings (Rule Checker)")
                
                # Top metrics columns
                cols = st.columns(min(len(doc["counts"]), 4))
                for i, (k, v) in enumerate(sorted(doc["counts"].items(), key=lambda kv: -kv[1])):
                    cols[i % len(cols)].metric(k, v)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Table & Chart Row
                col_table, col_chart = st.columns([3, 2])
                
                with col_table:
                    st.markdown("###### Detailed Matches")
                    display_findings = []
                    for f in doc["findings"]:
                        safe_f = f.copy()
                        safe_f["value"] = mask_value(f["type"], f["value"])
                        display_findings.append({
                            "Type": safe_f["type"],
                            "Value (Masked)": safe_f["value"],
                            "Location Span": f"Index {f['start']}-{f['end']}",
                            "Severity": "High" if f["type"] in st.session_state.high_risk_types else "Medium" if f["type"] in st.session_state.medium_risk_types else "Low"
                        })
                    st.dataframe(pd.DataFrame(display_findings), use_container_width=True, hide_index=True)
                    
                with col_chart:
                    st.markdown("###### Type Distribution")
                    chart_data = pd.DataFrame([
                        {"Sensitive Field": k, "Occurrences": v} 
                        for k, v in doc["counts"].items()
                    ])
                    bar = alt.Chart(chart_data).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                        y=alt.Y("Sensitive Field", sort="-x", axis=alt.Axis(labelAngle=0)),
                        x="Occurrences",
                        color=alt.Color(field="Sensitive Field", legend=None),
                        tooltip=["Sensitive Field", "Occurrences"]
                    ).properties(height=240)
                    st.altair_chart(bar, use_container_width=True)
            else:
                st.success("🎉 No sensitive PII/PCI-DSS data patterns detected in this document.")
    else:
        st.info("👋 Welcome! Please upload one or more documents above to generate insights and summaries.")

elif nav == "💬 Chatbot":
    if st.session_state.registry:
        if active_file:
            doc = st.session_state.registry[active_file]
            
            # Header Columns: Title vs Export Button
            col_title, col_export = st.columns([3, 1])
            with col_title:
                st.markdown(f"#### 💬 Chatting with **{active_file}**")
                st.caption("Secure document Q&A. The chatbot queries the in-memory RAG index and displays sources.")
            with col_export:
                if doc["chat_history"]:
                    chat_txt = compile_chat_history(doc["chat_history"])
                    st.markdown("<div style='margin-top: 6px;'></div>", unsafe_allow_html=True)
                    st.download_button(
                        label="💾 Export Chat (.txt)",
                        data=chat_txt,
                        file_name=f"chat_history_{doc['filename']}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            st.markdown("---")
            
            # Display chat messages with custom avatars and sources
            for message in doc["chat_history"]:
                role = message["role"]
                avatar = "👤" if role == "user" else "🛡️"
                with st.chat_message(role, avatar=avatar):
                    st.markdown(message["content"])
                    if "sources" in message and message["sources"]:
                        st.markdown("---")
                        st.markdown("📂 **Sources & References:**")
                        for idx, src in enumerate(message["sources"]):
                            st.markdown(f"**Source {idx+1}:** *\"{src.strip()}\"*")
            
            # Chat input
            if query := st.chat_input("Ask a question about the active document..."):
                with st.chat_message("user", avatar="👤"):
                    st.markdown(query)
                doc["chat_history"].append({"role": "user", "content": query})
                
                with st.spinner("Retrieving document references and composing response..."):
                    response, sources = answer_question(
                        document_text=doc["text"],
                        counts=doc["counts"],
                        risk=doc["risk"],
                        risk_reasons=doc["meta"]["reasons"],
                        question=query,
                        chat_history=doc["chat_history"],
                        rag=doc["rag"]
                    )
                    
                    with st.chat_message("assistant", avatar="🛡️"):
                        st.markdown(response)
                        if sources:
                            st.markdown("---")
                            st.markdown("📂 **Sources & References:**")
                            for idx, src in enumerate(sources):
                                st.markdown(f"**Source {idx+1}:** *\"{src.strip()}\"*")
                                    
                doc["chat_history"].append({
                    "role": "assistant", 
                    "content": response, 
                    "sources": sources
                })
                
                audit_log.log_event("question_asked", doc["filename"], {"question": query})
                st.rerun()
    else:
        st.info("Please upload and scan a document to enable the Chatbot.")

elif nav == "📜 Audit History":
    st.markdown("### 📜 Compliance Audit Ledger")
    st.write("Access the interactive historical log of all scans, Q&A logs, and compliance activities.")
    
    # Read logs
    df = audit_log.get_logs()
    
    if df.empty:
        st.info("No audit logs found. Scanned documents and chat logs will be archived here.")
    else:
        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df_sorted = df.sort_values(by="timestamp", ascending=False)
        
        # Calculations for metrics
        scanned_count = len(df[df["event"] == "document_scanned"])
        asked_count = len(df[df["event"] == "question_asked"])
        redacted_count = len(df[df["event"] == "redaction_downloaded"])
        
        # Display Metric cards
        m1, m2, m3 = st.columns(3)
        m1.metric("Scanned Documents", scanned_count)
        m2.metric("Questions Resolved", asked_count)
        m3.metric("Downloads Exported", redacted_count)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visualization Row
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("##### Activity Timeline")
            df_time = df.copy()
            df_time["Date"] = df_time["timestamp"].dt.strftime("%Y-%m-%d")
            time_data = df_time.groupby(["Date", "event"]).size().reset_index(name="Count")
            
            line_chart = alt.Chart(time_data).mark_line(point=True).encode(
                x="Date:T",
                y="Count:Q",
                color=alt.Color("event:N", legend=alt.Legend(title="Event Type")),
                tooltip=["Date", "event", "Count"]
            ).properties(height=240)
            st.altair_chart(line_chart, use_container_width=True)
            
        with col2:
            st.markdown("##### Event Type Breakdown")
            event_counts = df.groupby("event").size().reset_index(name="Occurrences")
            bar_chart = alt.Chart(event_counts).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                y=alt.Y("event:N", sort="-x", title="Event Type"),
                x=alt.X("Occurrences:Q", title="Counts"),
                color=alt.Color("event:N", legend=None),
                tooltip=["event", "Occurrences"]
            ).properties(height=240)
            st.altair_chart(bar_chart, use_container_width=True)
            
        # Interactive Search & Table Section
        st.markdown("---")
        st.markdown("##### 🔍 Historic Logs Explorer")
        
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            events_avail = sorted(list(df["event"].unique()))
            selected_events = st.multiselect(
                "Filter by Event Type:",
                options=events_avail,
                default=events_avail
            )
        with col_f2:
            search_query = st.text_input("Search logs by Filename:")
            
        # Apply filters
        filtered_df = df_sorted[df_sorted["event"].isin(selected_events)]
        if search_query:
            filtered_df = filtered_df[filtered_df["filename"].str.contains(search_query, case=False, na=False)]
            
        # Clean up column displays for the UI
        display_df = filtered_df.copy()
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
