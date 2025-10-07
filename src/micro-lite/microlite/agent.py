# microlite/agent.py
from .llm import BaseGenerator
from .registry import FunctionRegistry
from .templates import render_prompt

class Agent:
    def __init__(self, generator: BaseGenerator, registry: FunctionRegistry):
        self.generator = generator
        self.registry = registry

    def run(self, user_input: str):
        funcs = self.registry.list_specs()
        prompt = render_prompt(user_input, funcs)
        result = self.generator.generate(prompt, functions=funcs)
        if result["type"] == "function_call":
            name = result["name"]
            args = result.get("args","")
            try:
                output = self.registry.call(name, args)
                return {"type":"function_result", "value": output}
            except Exception as e:
                return {"type":"error", "error": str(e)}
        else:
            return {"type":"text", "text": result.get("text","")}
