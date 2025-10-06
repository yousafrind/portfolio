import os, hashlib, time
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from ocr import image_to_text
from vision import analyse_image

CHROMA_DIR = "chroma"
DATA_DIR   = "data"
os.makedirs(CHROMA_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)

embedding = SentenceTransformerEmbeddings(model_name=os.getenv("EMBED_MODEL"))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def load_single(path: str):
    ext = path.lower()
    if ext.endswith(".pdf"):
        return PyPDFLoader(path).load()
    elif ext.endswith((".jpg", ".jpeg", ".png")):
        with open(path, "rb") as f:
            img_bytes = f.read()
        ocr_text  = image_to_text(img_bytes)
        vision    = analyse_image(img_bytes)
        fake_doc  = type("obj", (object,), {
            "page_content": f"OCR:\n{ocr_text}\n\nVision analysis:\n{vision}",
            "metadata": {"source": path, "ocr": True, "vision": True}
        })()
        return [fake_doc]
    elif ext.endswith(".docx"):
        return Docx2txtLoader(path).load()
    else:  # txt
        return TextLoader(path, encoding="utf-8").load()

def ingest(filenames: list):
    all_docs = []
    for fname in filenames:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)
        docs = load_single(fpath)
        splits = text_splitter.split_documents(docs)
        for doc in splits:
            doc.metadata["sha256"] = hashlib.sha256(fpath.encode()).hexdigest()
        all_docs.extend(splits)

    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
    db.add_documents(all_docs)
    db.persist()
    return len(all_docs)