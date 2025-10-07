import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    EMBEDDER = os.getenv("EMBEDDER", "local")  # local | openai
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    CHROMA_DIR = os.getenv("CHROMA_DIR", "./.chroma")
    TOP_K = int(os.getenv("TOP_K", "10"))
