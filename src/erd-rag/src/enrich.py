import re
from typing import Dict, Any, List

ABBR = {"cust": "customer", "addr": "address", "amt": "amount", "qty": "quantity", "dt": "date", "usr": "user", "prod": "product"}

def split_identifier(name: str) -> List[str]:
    name = name.replace("_"," ")
    # split camelCase as well
    parts = re.findall(r"[A-Z]?[a-z]+|[0-9]+", name)
    if not parts:
        parts = [name]
    return [p.lower() for p in parts]

def expand_parts(parts):
    return [ABBR.get(p, p) for p in parts]

def canonicalize(name: str) -> str:
    parts = split_identifier(name)
    parts = expand_parts(parts)
    return " ".join(parts)

def make_table_doc(table: Dict[str, Any]) -> Dict[str, str]:
    """
    Return a text doc for the table-level and per-column docs.
    """
    canonical = canonicalize(table["name"])
    col_lines = []
    col_docs = []
    for c in table["columns"]:
        cname = canonicalize(c["name"])
        col_line = f"{c['name']} ({c.get('type')}) - {c.get('desc','')}"
        col_lines.append(col_line)
        col_docs.append({
            "id": f"{table['name']}::{c['name']}",
            "text": f"Column `{c['name']}` (canonical: {cname}). Type: {c.get('type')}. Description: {c.get('desc','')}",
            "meta": {"table": table["name"], "column": c["name"], "canonical_column": cname}
        })

    table_text = (
        f"Table `{table['name']}` (canonical: {canonical}). "
        f"Description: {table.get('description','')}\nColumns:\n" +
        "\n".join(f"- {l}" for l in col_lines)
    )
    table_doc = {"id": f"table::{table['name']}", "text": table_text, "meta": {"table": table["name"], "canonical_table": canonical}}
    # relations
    rel_docs = []
    for r in table.get("relations", []):
        rel_docs.append({
            "id": f"rel::{table['name']}::{r['from_col']}->{r['to_table']}.{r['to_col']}",
            "text": f"Relation: `{table['name']}.{r['from_col']}` -> `{r['to_table']}.{r['to_col']}`",
            "meta": {"from_table": table["name"], "from_col": r['from_col'], "to_table": r['to_table'], "to_col": r['to_col']}
        })
    return {"table_doc": table_doc, "col_docs": col_docs, "rel_docs": rel_docs}
