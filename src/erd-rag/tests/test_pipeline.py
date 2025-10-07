import subprocess
import os
from src.utils import load_text
from src.parser import parse_markdown_erd
from src.enrich import make_table_doc

def test_parse_and_enrich():
    md = load_text("src/sample_erd.md")
    tables = parse_markdown_erd(md)
    assert "customers" in tables
    # create docs
    enriched = make_table_doc(tables["customers"])
    assert "customers" in enriched["table_doc"]["id"]
    assert any(c["id"].startswith("customers::") for c in enriched["col_docs"])
