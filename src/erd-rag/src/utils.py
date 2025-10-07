import json
from typing import Any

def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf8") as f:
        return f.read()
