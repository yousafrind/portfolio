from chromadb.utils import embedding_functions
import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict
from config import Config

class VectorIndexer:
    def __init__(self, embedder):
        persist = Config.CHROMA_DIR
        os.makedirs(persist, exist_ok=True)
        self.client = chromadb.Client(Settings(chroma_db_impl="chromadb.db.sqlite3", persist_directory=persist))
        # choose embedding function wrapper only for chroma when using local sbert we will pass raw vectors
        self.collection = self.client.get_or_create_collection(name="erd_docs")
        self.embedder = embedder

    def upsert(self, ids: List[str], texts: List[str], metadatas: List[Dict]):
        if Config.EMBEDDER == "local":
            # Chroma can accept embeddings directly; but to keep simple, we'll store vectors in the collection via embeddings param
            vectors = self.embedder.embed(texts)
            self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors)
        else:
            # openai or remote: store text only and let chroma call embedding function? We'll embed first then store
            vectors = self.embedder.embed(texts)
            self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors)

    def query_vectors(self, vector, n_results=10):
        # query by embedding
        res = self.collection.query(query_embeddings=[vector], n_results=n_results, include=["metadatas","distances","documents","ids"])
        return res

    def query_text(self, text, n_results=10):
        # fallback to text search
        res = self.collection.query(query_texts=[text], n_results=n_results, include=["metadatas","distances","documents","ids"])
        return res
