import re
from typing import Dict, Any, List

def parse_markdown_erd(md_text: str) -> Dict[str, Any]:
    """
    Parses a markdown ERD with sections like:
    ### Table: orders
    Description: ...
    - `order_id` (INT) - primary key
    - `cust_id` (INT) - FK -> customers.id
    """
    tables = {}
    cur = None
    for line in md_text.splitlines():
        line = line.rstrip()
        m_table = re.match(r"^###\s*Table\s*:\s*(\S+)", line, re.I)
        if m_table:
            cur = m_table.group(1).strip()
            tables[cur] = {"name": cur, "description": "", "columns": [], "relations": []}
            continue
        if cur is None:
            continue
        m_col = re.match(r"^\s*-\s*`?([\w_]+)`?\s*(\([^)]+\))?\s*-\s*(.*)", line)
        if m_col:
            col, typ, desc = m_col.groups()
            tables[cur]["columns"].append({
                "name": col,
                "type": typ.strip("()") if typ else None,
                "desc": desc.strip()
            })
            # detect FK relations in desc
            fk = re.search(r"FK\s*->\s*([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)", desc)
            if fk:
                tables[cur]["relations"].append({"from_col": col, "to_table": fk.group(1), "to_col": fk.group(2)})
            continue
        m_desc = re.match(r"^\s*Description\s*:\s*(.*)", line, re.I)
        if m_desc:
            tables[cur]["description"] += (" " + m_desc.group(1).strip())
    return tables
