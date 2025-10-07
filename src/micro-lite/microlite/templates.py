# microlite/templates.py
from typing import List

PROMPT_TEMPLATE = """You are a helpful agent. Here are available functions:
{functions}

User: {user_input}

If you want to run a function, reply starting with "CALL:<fn_name> <args...>" otherwise reply normally.
"""

def render_prompt(user_input: str, functions: List[dict]) -> str:
    func_lines = []
    for f in functions:
        func_lines.append(f"- {f['name']} {f['signature']}: {f['description']}")
    return PROMPT_TEMPLATE.format(functions="\n".join(func_lines), user_input=user_input)
