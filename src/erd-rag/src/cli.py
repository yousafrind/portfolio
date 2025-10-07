import argparse
from parser import parse_markdown_erd
from enrich import make_table_doc
from embedder import Embedder
from indexer import VectorIndexer
from retriever import Retriever
from config import Config
from utils import load_text
import os
from tqdm import tqdm

def build_index(erd_path: str):
    md = load_text(erd_path)
    tables = parse_markdown_erd(md)
    # create docs
    docs = []
    for tname, t in tables.items():
        enriched = make_table_doc(t)
        docs.append(enriched["table_doc"])
        docs.extend(enriched["col_docs"])
        docs.extend(enriched["rel_docs"])

    # build embedding + vector store
    emb = Embedder()
    indexer = VectorIndexer(emb)
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metas = [d["meta"] for d in docs]
    print(f"Upserting {len(ids)} docs to vector store...")
    indexer.upsert(ids, texts, metas)

    # build sparse and graph for retriever
    retr = Retriever(emb, indexer)
    retr.ingest_docs_for_sparse(docs)
    # build graph from relations
    # read relations from parsed tables
    tables_list = list(tables.keys())
    retr.ingest_graph(tables_list)
    for t in tables.values():
        for r in t.get("relations", []):
            retr.graph.add_relation(t["name"], r["to_table"])

    # persist a minimal metadata file
    import json
    with open(os.path.join(Config.CHROMA_DIR, "meta_tables.json"), "w") as f:
        json.dump({"tables": tables_list}, f, indent=2)
    print("Index built and metadata persisted.")

def query_index(q: str):
    emb = Embedder()
    indexer = VectorIndexer(emb)
    retr = Retriever(emb, indexer)
    # load sparse from chroma collection documents (for simplicity read all docs)
    col = indexer.collection
    all_data = col.get(include=["ids","documents","metadatas"])
    docs = []
    for i, did in enumerate(all_data["ids"]):
        docs.append({"id": did, "text": all_data["documents"][i], "meta": all_data["metadatas"][i]})
    retr.ingest_docs_for_sparse(docs)
    # graph: reconstruct small graph from meta if present
    try:
        import json, os
        with open(os.path.join(Config.CHROMA_DIR, "meta_tables.json")) as f:
            meta = json.load(f)
            retr.ingest_graph(meta.get("tables", []))
    except Exception:
        pass

    results = retr.hybrid_query(q, top_k=Config.TOP_K)
    from rag import build_rag_prompt
    prompt = build_rag_prompt(q, results)
    print("=== TOP RESULTS ===")
    for r in results[:10]:
        print(f"- {r['id']} (score={r['score']:.3f}) -> {r['meta']}")
    print("\n=== RAG PROMPT ===\n")
    print(prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    b.add_argument("--erd", required=True)
    q = sub.add_parser("query")
    q.add_argument("--q", required=True)
    args = parser.parse_args()
    if args.cmd == "build":
        build_index(args.erd)
    elif args.cmd == "query":
        query_index(args.q)
    else:
        parser.print_help()
