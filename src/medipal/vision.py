from transformers import pipeline
from PIL import Image
import io, os, logging
+
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("VISION_MODEL")

# try to initialize a local classifier; fail gracefully if the environment
# doesn't have the model or transformers isn't fully available.
clf = None
try:
    clf = pipeline("image-classification", model=MODEL_NAME, device=-1)  # CPU
except Exception as e:
    logger.debug("Local vision model unavailable: %s", e)


def _google_vision_labels(file_bytes: bytes, max_results: int = 5) -> list:
    """Use Google Cloud Vision label detection as a fallback."""
    try:
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=file_bytes)
        resp = client.label_detection(image=image, max_results=max_results)
        if getattr(resp, "error", None) and getattr(resp.error, "message", None):
            logger.debug("Google Vision label error: %s", resp.error.message)
            return []
        labels = []
        for lbl in getattr(resp, "label_annotations", []):
            labels.append({"label": lbl.description, "score": round(lbl.score, 3)})
        return labels
    except Exception as e:
        logger.debug("Google Vision labels unavailable or failed: %s", e)
        return []


def analyse_image(file_bytes: bytes) -> dict:
    """Return predictions for an image. Try local classifier first, fallback to
    Google Cloud Vision label detection if needed.
    """
    # local pipeline
    try:
        if clf:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            preds = clf(image, top_k=3)
            return {"predictions": [{"label": p.get("label") or p["label"], "score": round(p.get("score", 0), 3)} for p in preds]}
    except Exception as e:
        logger.debug("Local vision model failed: %s", e)

    # fallback to Google Vision labels
    labels = _google_vision_labels(file_bytes)
    return {"predictions": labels}