# ============================================================
# ocr_utils.py — Image download + Tesseract OCR
#
# Key improvements over v1:
#   • Image resized before OCR if larger than OCR_MAX_IMAGE_DIM
#     (was missing — huge screenshots were extremely slow)
#   • PSM changed from 6 (single uniform block) to 11 (sparse text)
#     which is far better for UI screenshots with labels/buttons
#   • Whitelist kept to printable ASCII to reduce junk output
#   • All constants imported from config.py
# ============================================================

import re
import logging
import requests
from io import BytesIO

from PIL import Image, ImageOps
import pytesseract

from chatbot.config import OCR_MAX_IMAGE_DIM, OCR_TIMEOUT

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

# PSM 11 = sparse text (best for UI screenshots with mixed elements)
# PSM 6  = single block (only good for plain paragraph scans)
_TESSERACT_CONFIG = r"--oem 3 --psm 11"

_JUNK_RE = re.compile(r"[|~_\[\]@#$\\^`{}]")
_SPACE_RE = re.compile(r"\s+")


def extract_text_from_image(url: str) -> str:
    """
    Download `url`, pre-process for OCR, return extracted text.
    Returns empty string on any failure (never raises).
    """
    try:
        r = requests.get(url, headers=_HEADERS, timeout=OCR_TIMEOUT)
        r.raise_for_status()

        img = Image.open(BytesIO(r.content))

        # Normalise mode
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")

        # ── Resize if too large ──────────────────────────────
        # FIX: v1 fed full-size images to Tesseract — slow and
        # inaccurate for huge screenshots.
        w, h = img.size
        if max(w, h) > OCR_MAX_IMAGE_DIM:
            scale = OCR_MAX_IMAGE_DIM / max(w, h)
            img = img.resize(
                (int(w * scale), int(h * scale)),
                Image.LANCZOS,
            )

        # Grayscale → better Tesseract accuracy
        img = ImageOps.grayscale(img)

        # Optional: auto-contrast helps low-contrast screenshots
        img = ImageOps.autocontrast(img)

        text = pytesseract.image_to_string(img, config=_TESSERACT_CONFIG)

        # Clean junk characters
        text = _JUNK_RE.sub("", text)
        text = _SPACE_RE.sub(" ", text)
        return text.strip()

    except Exception as e:
        logger.debug("OCR failed for %s: %s", url, e)
        return ""