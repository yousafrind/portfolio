from typing import List, Dict

RAG_TEMPLATE = """SCHEMA CONTEXT:
{schema_block}

USER QUESTION:
{question}

INSTRUCTIONS:
Using only the SCHEMA CONTEXT above, answer the USER QUESTION. When referencing a table or column, include the exact table/column names from the schema and a brief justification (which columns or relations you used). If you can't answer, say you don't know.
"""

def build_schema_block(docs: List[Dict]) -> str:
    """
    Build a compact schema block from retrieved docs (tables/columns/relations)
    """
    lines = []
    # group by table
    tables = {}
    for d in docs:
        meta = d.get("meta", {})
        # table entries
        if meta.get("table"):
            tname = meta["table"]
            tables.setdefault(tname, {"table_doc": d["text"], "columns": [], "relations": []})
        # column docs
        if meta.get("column") and meta.get("table"):
            tables.setdefault(meta["table"], {"table_doc": "", "columns": [], "relations": []})["columns"].append(d["text"])
        # relations
        if meta.get("from_table") and meta.get("to_table"):
            tables.setdefault(meta["from_table"], {"table_doc": "", "columns": [], "relations": []})["relations"].append(d["text"])
    for t, v in tables.items():
        lines.append(f"- Table: {t}")
        if v["table_doc"]:
            # include first line only
            lines.append(f"  {v['table_doc'].splitlines()[0]}")
        if v["columns"]:
            lines.append("  Columns:")
            for c in v["columns"][:10]:
                lines.append(f"    - {c}")
        if v["relations"]:
            lines.append("  Relations:")
            for r in v["relations"]:
                lines.append(f"    - {r}")
    return "\n".join(lines)

def build_rag_prompt(question: str, docs: List[Dict]) -> str:
    schema_block = build_schema_block(docs)
    return RAG_TEMPLATE.format(schema_block=schema_block, question=question)
