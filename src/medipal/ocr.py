import easyocr, io, logging, os
from PIL import Image

logger = logging.getLogger(__name__)

# initialize EasyOCR reader if possible; keep gracefully handling failures so the
# module can be imported even on systems without GPU or with missing deps.
reader = None
try:
    reader = easyocr.Reader(['en'], gpu=False)
except Exception:
    logger.debug("EasyOCR not available or failed to init; will try Google Vision as fallback")


def _google_vision_ocr(file_bytes: bytes) -> str:
    """Try Google Cloud Vision OCR as a fallback when EasyOCR fails or returns nothing.

    Requires Google Cloud credentials (GOOGLE_APPLICATION_CREDENTIALS) or a GEMINI_API_KEY
    environment variable to be set so the Google client libraries can authenticate.
    """
    try:
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=file_bytes)
        resp = client.text_detection(image=image)
        if getattr(resp, "error", None) and getattr(resp.error, "message", None):
            logger.debug("Google Vision OCR error: %s", resp.error.message)
            return ""
        texts = [t.description for t in getattr(resp, "text_annotations", [])]
        return texts[0].strip() if texts else ""
    except Exception as e:
        logger.debug("Google Vision OCR unavailable or failed: %s", e)
        return ""


def image_to_text(file_bytes: bytes) -> str:
    """Return extracted text from an image. Try local EasyOCR first, then Google Vision.

    Returns an empty string if no OCR result was found.
    """
    # 1) try EasyOCR if initialized
    try:
        if reader:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            result = reader.readtext(image, detail=0)
            text = " ".join(result).strip()
            if text:
                return text
    except Exception as e:
        logger.debug("EasyOCR processing failed: %s", e)

    # 2) fallback to Google Cloud Vision (if credentials present)
    google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GEMINI_API_KEY")
    if google_creds:
        text = _google_vision_ocr(file_bytes)
        if text:
            return text

    # final fallback: empty string
    logger.warning("OCR produced no text (EasyOCR + Google Vision fallback exhausted)")
    return ""