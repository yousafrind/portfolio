import os, requests, hashlib, logging
from langchain.llms import LlamaCpp

logger = logging.getLogger(__name__)

MODEL_DIR = "models"
MODEL_URL = os.getenv("MED_MODEL_URL")
MODEL_PATH = os.path.join(MODEL_DIR, "medalpaca-7b.Q4_K_M.gguf")


def download_once():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        if not MODEL_URL:
            raise RuntimeError("MED_MODEL_URL is not set; cannot download MedAlpaca model")
        print("Downloading MedAlpaca-7B quantized (≈ 4 GB) …")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return MODEL_PATH


def get_med_llm():
    """Return a local LlamaCpp instance. This may raise if the model isn't available."""
    return LlamaCpp(
        model_path=download_once(),
        n_ctx=2048,
        temperature=0.3,
        max_tokens=512,
        n_threads=8,           # adjust to your CPU
    )


def get_resilient_llm():
    """Prefer a local MedAlpaca LLM; if it cannot be created, fall back to
    OpenAI or Google Gemini (when configured). Returns an LLM object or None.
    """
    # 1) try local MedAlpaca
    try:
        return get_med_llm()
    except Exception as e:
        logger.debug("Local Med LLM unavailable: %s", e)

    # 2) try OpenAI (if key present)
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain.llms import OpenAI
            return OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.3)
        except Exception as e:
            logger.debug("OpenAI init failed: %s", e)

    # 3) try Gemini (Google) if credentials set
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        except Exception as e:
            logger.debug("Gemini init failed: %s", e)

    return None