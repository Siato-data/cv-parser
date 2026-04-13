import io
import logging
from pathlib import Path

import pdfplumber
import fitz
import docx

logger = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".docx", ".txt"}
MIN_TEXT_LENGTH = 100


def extract(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Format non supporté: {ext}. Acceptés: pdf, docx, txt")

    if ext == ".pdf":
        text = _pdfplumber(file_bytes)
        if len(text.strip()) < MIN_TEXT_LENGTH:
            logger.warning(f"{filename}: pdfplumber trop court, fallback pymupdf")
            text = _pymupdf(file_bytes)
    elif ext == ".docx":
        text = _docx(file_bytes)
    elif ext == ".txt":
        text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        raise ValueError(f"Impossible d'extraire du texte de {filename}.")

    return text.strip()


def _pdfplumber(b: bytes) -> str:
    parts = []
    try:
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            for page in pdf.pages:
                t = page.extract_text(x_tolerance=2, y_tolerance=2)
                if t:
                    parts.append(t)
    except Exception as e:
        logger.warning(f"pdfplumber error: {e}")
    return "\n\n".join(parts)


def _pymupdf(b: bytes) -> str:
    parts = []
    try:
        doc = fitz.open(stream=b, filetype="pdf")
        for page in doc:
            parts.append(page.get_text("text"))
        doc.close()
    except Exception as e:
        logger.warning(f"pymupdf error: {e}")
    return "\n\n".join(parts)


def _docx(b: bytes) -> str:
    doc = docx.Document(io.BytesIO(b))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
