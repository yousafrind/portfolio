from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import networkx as nx
from tqdm import tqdm
from config import Config
import numpy as np

class SparseIndex:
    def __init__(self):
        self.corpus = []
        self.ids = []
        self.bm25 = None

    def add(self, text: str, id: str):
        tokens = text.lower().split()
        self.corpus.append(tokens)
        self.ids.append(id)

    def build(self):
        self.bm25 = BM25Okapi(self.corpus)

    def query(self, q: str, topk=10) -> List[Tuple[str,float]]:
        if self.bm25 is None:
            return []
        tokens = q.lower().split()
        scores = self.bm25.get_scores(tokens)
        idx = np.argsort(-scores)[:topk]
        results = [(self.ids[i], float(scores[i])) for i in idx if scores[i] > 0]
        return results

class GraphHelper:
    def __init__(self):
        self.G = nx.DiGraph()

    def add_table(self, table_name: str):
        self.G.add_node(table_name)

    def add_relation(self, from_table, to_table):
        self.G.add_edge(from_table, to_table)

    def one_hop_neighbors(self, table_names: List[str]) -> List[str]:
        out = set()
        for t in table_names:
            out.update(self.G.predecessors(t))
            out.update(self.G.successors(t))
        return list(out)

class Retriever:
    def __init__(self, embedder, indexer):
        self.embedder = embedder
        self.indexer = indexer
        self.sparse = SparseIndex()
        self.graph = GraphHelper()

    def ingest_docs_for_sparse(self, docs: List[Dict]):
        for d in docs:
            # use id and text
            self.sparse.add(d["id"], d["text"])
        self.sparse.build()

    def ingest_graph(self, tables):
        for t in tables:
            self.graph.add_table(t)
        # edges parsed from metadata in chroma later when building

    def hybrid_query(self, query: str, top_k:int=Config.TOP_K) -> List[Dict]:
        # dense:
        q_vec = self.embedder.embed([query])[0]
        dense_res = self.indexer.query_vectors(q_vec, n_results=top_k)
        dense_ids = dense_res["ids"][0]
        dense_docs = []
        for i, docid in enumerate(dense_ids):
            dense_docs.append({
                "id": docid,
                "score": 1.0 - dense_res["distances"][0][i]  # chroma distances are 0..1 (similarity) - adjust if needed
            })
        # sparse:
        sparse_res = self.sparse.query(query, topk=top_k)
        sparse_docs = [{"id": sid, "score": scr} for sid,scr in sparse_res]

        # combine scores (simple weighted)
        score_map = {}
        for d in dense_docs:
            score_map[d["id"]] = score_map.get(d["id"], 0) + 1.0 * d["score"]
        for s in sparse_docs:
            score_map[s["id"]] = score_map.get(s["id"], 0) + 0.8 * s["score"]

        # graph expansion: if any table doc hits, include neighbors with small boost
        table_hits = [k for k in score_map.keys() if k.startswith("table::")]
        tables = [k.split("::",1)[1] for k in table_hits]
        neighbors = self.graph.one_hop_neighbors(tables)
        for nb in neighbors:
            key = f"table::{nb}"
            score_map[key] = score_map.get(key, 0) + 0.5

        # fetch doc details from chroma for top ids
        sorted_ids = sorted(score_map.items(), key=lambda x: -x[1])[:top_k]
        ids_only = [i[0] for i in sorted_ids]
        # query chroma for exact ids
        # chroma supports get with ids
        docs = []
        col = self.indexer.collection
        if ids_only:
            fetch = col.get(ids=ids_only, include=["metadatas","documents"])
            for i, did in enumerate(fetch["ids"]):
                docs.append({
                    "id": did,
                    "text": fetch["documents"][i],
                    "meta": fetch["metadatas"][i],
                    "score": score_map.get(did, 0)
                })
        return docs
