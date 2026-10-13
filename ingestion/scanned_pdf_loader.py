"""
PDF loading for textbooks — native text, vision OCR, or automatic fallback.

Extracts machine-readable text from PDFs: uses embedded text when present; for poor
or scanned pages, rasterizes pages with PyMuPDF and transcribes them via the vision
API (optional preprocessing for cleaner Bangla). Controlled by ``PDF_INGESTION_MODE``
and related config; feeds ``Document`` streams into COPE chunking.
"""
from __future__ import annotations

import io
from typing import List

import fitz
from langchain_core.documents import Document
from PIL import Image, ImageEnhance, ImageOps

import config
from utils.llm_client import vision_page_transcription
from utils.logger import log_info
from utils.prompt_loader import load_prompt


def _normalize_mode(mode: str) -> str:
    """``ocr`` is kept as a backward-compatible alias for ``vision``."""
    m = (mode or "vision").strip().lower()
    if m == "ocr":
        return "vision"
    return m


def _render_page_to_png_bytes(page: fitz.Page, scale: float) -> bytes:
    """Rasterize one PDF page to PNG bytes (RGB)."""
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    img = _preprocess_page_image(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _preprocess_page_image(img: Image.Image) -> Image.Image:
    """Apply configurable image cleanup to improve scanned Bangla OCR quality."""
    if not bool(getattr(config, "PDF_PREPROCESS_ENABLE", True)):
        return img

    out = img
    if bool(getattr(config, "PDF_PREPROCESS_GRAYSCALE", True)):
        out = ImageOps.grayscale(out).convert("RGB")
    if bool(getattr(config, "PDF_PREPROCESS_AUTOCONTRAST", True)):
        out = ImageOps.autocontrast(out, cutoff=1)

    sharpen = float(getattr(config, "PDF_PREPROCESS_SHARPEN_FACTOR", 1.0))
    if sharpen > 1.0:
        out = ImageEnhance.Sharpness(out).enhance(sharpen)

    return out


def _native_page_text(page: fitz.Page) -> str:
    return (page.get_text("text") or "").strip()


_DEFAULT_VISION_SYSTEM = load_prompt("pdf_vision_system_prompt.md")
_DEFAULT_VISION_USER = load_prompt("pdf_vision_user_prompt.md")


def _transcribe_page_png(png_bytes: bytes) -> str:
    system_prompt = getattr(config, "PDF_VISION_SYSTEM_PROMPT", "") or _DEFAULT_VISION_SYSTEM
    user_instruction = getattr(config, "PDF_VISION_USER_PROMPT", "") or _DEFAULT_VISION_USER
    max_tokens = int(getattr(config, "PDF_VISION_MAX_TOKENS", 4096))
    detail = getattr(config, "PDF_VISION_DETAIL", "high")
    model = getattr(config, "PDF_VISION_MODEL", None)
    return vision_page_transcription(
        system_prompt,
        png_bytes,
        user_instruction=user_instruction,
        model=model,
        detail=str(detail),
        max_tokens=max_tokens,
        temperature=float(getattr(config, "PDF_VISION_TEMPERATURE", 0.0)),
    )


def load_pdf_documents(file_path: str) -> List[Document]:
    """
    Load a PDF as LangChain Documents (one per page).

    - ``native``: PyMuPDF text extraction only (selectable text PDFs).
    - ``vision`` (alias ``ocr``): rasterize each page → GPT-4o-mini vision transcription.
    - ``auto``: native text if the page has enough characters; otherwise vision.
    """
    mode = _normalize_mode(getattr(config, "PDF_INGESTION_MODE", "auto"))
    min_chars = int(getattr(config, "PDF_OCR_MIN_TEXT_CHARS", 40))
    scale = float(getattr(config, "PDF_RENDER_SCALE", 2.0))

    basename = file_path.split("\\")[-1].split("/")[-1]
    out: List[Document] = []

    doc = fitz.open(file_path)
    try:
        for i in range(len(doc)):
            page = doc[i]
            ingestion_method: str
            text: str

            if mode == "native":
                text = _native_page_text(page)
                ingestion_method = "native"

            elif mode == "vision":
                png = _render_page_to_png_bytes(page, scale)
                text = _transcribe_page_png(png)
                ingestion_method = "gpt4o_mini_vision"

            elif mode == "auto":
                native = _native_page_text(page)
                if len(native) >= min_chars:
                    text = native
                    ingestion_method = "native"
                else:
                    png = _render_page_to_png_bytes(page, scale)
                    text = _transcribe_page_png(png)
                    ingestion_method = "auto_gpt4o_mini_vision"
                if not text.strip() and native:
                    text = native
                    ingestion_method = "native_fallback"

            else:
                raise ValueError(
                    f"Unknown PDF_INGESTION_MODE: {mode!r} (use native|vision|ocr|auto)"
                )

            meta = {
                "source": basename,
                "page": i + 1,
                "pdf_ingestion": ingestion_method,
                "total_pages": len(doc),
            }
            out.append(Document(page_content=text, metadata=meta))
    finally:
        doc.close()

    log_info(
        f"PDF loaded {basename}: {len(out)} pages, mode={mode!r} "
        f"(vision uses {getattr(config, 'PDF_VISION_MODEL', config.OPENAI_MODEL)})"
    )
    return out
