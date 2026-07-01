import json
import os
import datetime
import threading
import pandas as pd
from typing import Dict, Any, List

LOG_FILE = "audit_log.jsonl"
log_lock = threading.Lock()

def log_event(event_type: str, filename: str, metadata: Dict[str, Any]) -> None:
    """
    Appends a structured event to the audit log in a thread-safe manner.
    """
    try:
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": event_type,
            "filename": filename,
            **metadata
        }
        
        with log_lock:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"[Audit Log] Failed to write log: {e}")

def get_logs() -> pd.DataFrame:
    """
    Reads the JSONL log file and returns it as a pandas DataFrame.
    """
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(columns=["timestamp", "event", "filename"])
        
    try:
        entries = []
        with log_lock:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            pass
                            
        df = pd.DataFrame(entries)
        # Reorder columns so timestamp, event, and filename are first
        all_cols = list(df.columns)
        standard_cols = ["timestamp", "event", "filename"]
        for c in standard_cols:
            if c in all_cols:
                all_cols.remove(c)
        df = df[standard_cols + all_cols]
        return df
    except Exception as e:
        print(f"[Audit Log] Failed to read logs: {e}")
        return pd.DataFrame(columns=["timestamp", "event", "filename"])
