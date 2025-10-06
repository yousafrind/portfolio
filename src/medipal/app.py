from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import os, time, json, logging

from llm_med import get_resilient_llm, MODEL_PATH
from redact import redact_query, redact_prompt, redact_for_api
from search_tavily import search_evidence
from ingest import ingest, DATA_DIR, CHROMA_DIR
from ocr import image_to_text
from vision import analyse_image

logger = logging.getLogger(__name__)
app = FastAPI(title="MediPal")

# Vectorstore will be initialized on startup (lazy). We keep it on app.state.
@app.on_event("startup")
def startup_event():
    health = {}

    # Local MedAlpaca model file presence (won't trigger download)
    health["local_med_model_present"] = os.path.exists(MODEL_PATH)

    # Cloud provider credentials
    health["openai_api_key"] = bool(os.getenv("OPENAI_API_KEY"))
    health["gemini_api_key"] = bool(os.getenv("GEMINI_API_KEY"))
    health["google_application_credentials"] = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

    # EasyOCR import check
    try:
        import easyocr  # type: ignore
        health["easyocr_installed"] = True
    except Exception:
        health["easyocr_installed"] = False

    # Google Vision client availability
    try:
        import google.cloud.vision  # type: ignore
        health["google_vision_installed"] = True
    except Exception:
        health["google_vision_installed"] = False

    # Try to initialize Chroma/embeddings (lazy) — non-fatal if missing
    try:
        from langchain.embeddings import SentenceTransformerEmbeddings  # type: ignore
        from langchain.vectorstores import Chroma  # type: ignore
        emb = SentenceTransformerEmbeddings(model_name=os.getenv("EMBED_MODEL"))
        app.state.vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb)
        health["vectorstore_ready"] = True
    except Exception as e:
        logger.debug("Vectorstore init failed: %s", e)
        app.state.vectorstore = None
        health["vectorstore_ready"] = False

    app.state.health = health
    logger.info("Startup health: %s", json.dumps(health))


class ImageUploadResponse(BaseModel):
    filename: str
    ocr_text: str
    vision: dict
    chars: int
    ingest_ok: bool


class AskWeb(BaseModel):
    question: str
    use_web: bool = False


@app.get("/health", summary="Service health and available backends")
def health_check():
    return app.state.health if hasattr(app.state, "health") else {"status": "starting"}


@app.post("/upload-image", summary="Photo / scan → OCR + vision summary → ingest")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, "Only JPEG/PNG")
    content = await file.read()
    ocr_text = image_to_text(content)
    vision = analyse_image(content)
    if not ocr_text and not vision.get("predictions"):
        raise HTTPException(422, "No text or vision insight found")
    fname = f"ocr_{int(time.time())}.txt"
    fpath = os.path.join(DATA_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(f"OCR:\n{ocr_text}\n\nVision:\n{json.dumps(vision, indent=2)}")
    chunks = ingest([fname])
    return ImageUploadResponse(
        filename=fname, ocr_text=ocr_text[:200], vision=vision, chars=len(ocr_text), ingest_ok=chunks > 0
    )


@app.post("/ask-web", summary="RAG + optional internet evidence")
def ask_web(body: AskWeb):
    if not getattr(app.state, "vectorstore", None):
        raise HTTPException(503, "Vectorstore not ready; ensure embeddings/chroma are installed and initialized")
    tic = time.perf_counter()
    docs = app.state.vectorstore.similarity_search(body.question, k=4)
    local_ctx = "\n".join([d.page_content for d in docs])
    web_query = redact_query(body.question) if body.use_web else ""
    if body.use_web:
        # use the redacted query for external search
        web_ctx = search_evidence(web_query)
    else:
        web_ctx = ""
    context = f"User record:\n{local_ctx}\n\nInternet evidence:\n{web_ctx}".strip()
    prompt = f"Answer the medical question briefly.\nContext:\n{context}\n\nQ: {body.question}\nA:"
    # redact prompt before sending to cloud LLMs
    redacted_prompt, redact_counts = redact_for_api(prompt)
    if redact_counts:
        logger.info("Redaction applied before LLM call: %s", redact_counts)
    llm = get_resilient_llm()
    if not llm:
        raise HTTPException(503, "No LLM available; configure MED_MODEL_URL, OPENAI_API_KEY or GEMINI_API_KEY")
    try:
        # if llm is local (MedAlpaca LlamaCpp) send original prompt; for remote
        # providers we prefer to send the redacted prompt
        if hasattr(llm, "__call__") and llm.__class__.__module__.startswith("llama"):
            answer = llm(prompt)
        else:
            # cloud LLMs
            answer = llm(redacted_prompt) if hasattr(llm, "__call__") else llm.invoke(redacted_prompt).content
    except Exception:
        logger.exception("LLM call failed")
        raise HTTPException(500, "LLM call failed")
    toc = time.perf_counter()
    tokens = len(prompt.split()) + len(answer.split())
    return {"answer": answer, "latency_sec": round(toc - tic, 3), "tokens": tokens,
            "sources": [d.metadata.get("source", "?") for d in docs], "web_used": body.use_web}
from llm_med import get_med_llm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import OpenAI
from search_tavily import search_evidence
from fastapi import File, UploadFile, HTTPException
from pydantic import BaseModel
import os, time, json

def get_llm():
    provider = os.getenv("LLM_PROVIDER")
    if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    elif provider == "openai" and os.getenv("OPENAI_API_KEY"):
        return OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.3)
    else:  # free CPU
        return get_resilient_llm()
    """Get a working LLM instance. Preference: local MedAlpaca -> configured provider (openai) -> Gemini (if available). Returns an object or raises HTTPException.
    """
    # 1) local/resilient LLM (may return ChatGoogleGenerativeAI if local fails)
    llm = get_resilient_llm()
    if llm:
        return llm

    # 2) explicit OpenAI provider if configured
    if os.getenv("OPENAI_API_KEY"):
        return OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.3)

    # 3) Gemini fallback if configured
    if os.getenv("GEMINI_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        except Exception:
            pass

    raise HTTPException(503, "No LLM available; configure MED_MODEL_URL, OPENAI_API_KEY or GEMINI_API_KEY")



class ImageUploadResponse(BaseModel):
    filename: str
    ocr_text: str
    vision: dict
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from pydantic import BaseModel
    import os, time, json, logging

    from llm_med import get_resilient_llm
    from search_tavily import search_evidence
    from ingest import ingest, DATA_DIR, CHROMA_DIR
    from ocr import image_to_text
    from vision import analyse_image

    from langchain.vectorstores import Chroma
    from langchain.embeddings import SentenceTransformerEmbeddings

    logger = logging.getLogger(__name__)
    app = FastAPI(title="MediPal")

    # prepare vectorstore handle (lazy – will open existing DB)
    emb = SentenceTransformerEmbeddings(model_name=os.getenv("EMBED_MODEL"))
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb)


    class ImageUploadResponse(BaseModel):
        filename: str
        ocr_text: str
        vision: dict
        chars: int
        ingest_ok: bool


    class AskWeb(BaseModel):
        question: str
        use_web: bool = False


    @app.post("/upload-image", summary="Photo / scan → OCR + vision summary → ingest")
    async def upload_image(file: UploadFile = File(...)):
        if file.content_type not in ("image/jpeg", "image/png"):
            raise HTTPException(400, "Only JPEG/PNG")
        content = await file.read()
        ocr_text = image_to_text(content)
        vision = analyse_image(content)
        if not ocr_text and not vision.get("predictions"):
            raise HTTPException(422, "No text or vision insight found")
        fname = f"ocr_{int(time.time())}.txt"
        fpath = os.path.join(DATA_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(f"OCR:\n{ocr_text}\n\nVision:\n{json.dumps(vision, indent=2)}")
        chunks = ingest([fname])
        return ImageUploadResponse(
            filename=fname, ocr_text=ocr_text[:200], vision=vision, chars=len(ocr_text), ingest_ok=chunks > 0
        )


    @app.post("/ask-web", summary="RAG + optional internet evidence")
    def ask_web(body: AskWeb):
        tic = time.perf_counter()
        docs = vectorstore.similarity_search(body.question, k=4)
        local_ctx = "\n".join([d.page_content for d in docs])
        web_ctx = search_evidence(body.question) if body.use_web else ""
        context = f"User record:\n{local_ctx}\n\nInternet evidence:\n{web_ctx}".strip()
        prompt = f"Answer the medical question briefly.\nContext:\n{context}\n\nQ: {body.question}\nA:"
        llm = get_resilient_llm()
        if not llm:
            raise HTTPException(503, "No LLM available; configure MED_MODEL_URL, OPENAI_API_KEY or GEMINI_API_KEY")
        # call LLM (support both callable and langchain-style objects)
        try:
            answer = llm(prompt) if hasattr(llm, "__call__") else llm.invoke(prompt).content
        except Exception as e:
            logger.exception("LLM call failed")
            raise HTTPException(500, "LLM call failed")
        toc = time.perf_counter()
        tokens = len(prompt.split()) + len(answer.split())
        return {"answer": answer, "latency_sec": round(toc - tic, 3), "tokens": tokens,
                "sources": [d.metadata.get("source", "?") for d in docs], "web_used": body.use_web}