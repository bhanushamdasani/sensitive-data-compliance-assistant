import io
import pandas as pd
from pypdf import PdfReader
from PIL import Image

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

def parse_txt(file_bytes: bytes) -> str:
    """
    Decodes raw file bytes into a UTF-8 text string, ignoring any encoding errors.
    """
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"[Error parsing TXT: {str(e)}]"

def parse_csv(file_bytes: bytes) -> str:
    """
    Reads CSV file bytes and lists columns and rows formatted as key-value pairs.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        if df.empty:
            return "[CSV is empty]"
            
        lines = ["Columns: " + ", ".join(str(c) for c in df.columns)]
        for _, row in df.iterrows():
            line = ", ".join(f"{col}: {row[col]}" for col in df.columns)
            lines.append(line)
        return "\n".join(lines)
    except Exception as e:
        return f"[Error parsing CSV: {str(e)}]"

def parse_pdf(file_bytes: bytes) -> str:
    """
    Extracts text page-by-page from a PDF document using PdfReader.
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)
    except Exception as e:
        return f"[Error parsing PDF: {str(e)}]"

def parse_image(file_bytes: bytes) -> str:
    """
    Extracts text from images (PNG, JPG) using Tesseract OCR if dependencies are present.
    """
    if not PYTESSERACT_AVAILABLE:
        return "[Error: The 'pytesseract' package is not installed in the Python environment. Run `pip install pytesseract` to scan images.]"
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        if "TesseractNotFoundError" in type(e).__name__:
            return "[Error: Tesseract OCR binary is not installed on the system path. Image scanning is unavailable.]"
        return f"[Error parsing Image: {str(e)}]"

def extract_text(filename: str, file_bytes: bytes) -> str:
    """
    Selects and runs the appropriate parser based on the file extension.
    """
    lower = filename.lower()
    
    if lower.endswith(".pdf"):
        return parse_pdf(file_bytes)
    if lower.endswith(".csv"):
        return parse_csv(file_bytes)
    if lower.endswith(".txt"):
        return parse_txt(file_bytes)
    if lower.endswith((".png", ".jpg", ".jpeg")):
        return parse_image(file_bytes)
        
    raise ValueError(f"Unsupported file type: {filename}. Use PDF, TXT, CSV, or Images.")