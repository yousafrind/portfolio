import os
import json
import requests
from typing import Any, Dict, List, Optional, Union

try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ------------------------------------------------------------
#  Helper: Convert registry metadata into JSON schemas
# ------------------------------------------------------------
def build_function_schemas(functions: List[dict]) -> List[Dict[str, Any]]:
    """
    Convert simple function specs into standardized JSON schema
    for LLM function calling (OpenAI-style).
    """
    schemas = []
    for f in functions:
        # crude signature parser — in production, use inspect + pydantic
        schemas.append({
            "name": f["name"],
            "description": f.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": {
                    # you can parse this from the registry signature if detailed
                    "input": {"type": "string", "description": "Raw input text or parameters."}
                },
                "required": []
            }
        })
    return schemas


# ------------------------------------------------------------
#  Base interface
# ------------------------------------------------------------
class BaseGenerator:
    """Abstract class for all LLM adapters."""
    def generate(self, prompt: str, functions: Optional[List[dict]] = None) -> Dict[str, Any]:
        raise NotImplementedError


# ------------------------------------------------------------
#  Ollama (local inference server)
# ------------------------------------------------------------
class OllamaGenerator(BaseGenerator):
    def __init__(self, model: str = "llama3", url: str = "http://localhost:11434/api/chat"):
        self.model = model
        self.url = url

    def generate(self, prompt: str, functions: Optional[List[dict]] = None) -> Dict[str, Any]:
        """
        Ollama doesn’t support native function calling yet, so we simulate
        a structured JSON response by prompting it directly.
        """
        schema = build_function_schemas(functions or [])
        sys_prompt = (
            "You are a function-calling AI. "
            "Respond ONLY in JSON of the form: "
            '{"type":"function_call","name":"...","args":{...}} or {"type":"text","text":"..."}.\n\n'
            f"Available functions: {json.dumps(schema)}"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        try:
            r = requests.post(self.url, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            # Ollama's response may be nested under message/content
            content = data.get("message", {}).get("content") or data.get("response", "")
            # Try to parse JSON from content
            try:
                result = json.loads(content)
                return result
            except Exception:
                return {"type": "text", "text": content.strip()}
        except Exception as e:
            return {"type": "error", "error": str(e)}


# ------------------------------------------------------------
#  OpenAI (function calling native)
# ------------------------------------------------------------
class OpenAIGenerator(BaseGenerator):
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        if not openai:
            raise ImportError("Please install openai: pip install openai")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self.model = model

    def generate(self, prompt: str, functions: Optional[List[dict]] = None) -> Dict[str, Any]:
        schema = build_function_schemas(functions or [])
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[{"type": "function", "function": s} for s in schema],
                tool_choice="auto"
            )

            msg = response.choices[0].message
            if msg.tool_calls:
                tool = msg.tool_calls[0]
                args = json.loads(tool.function.arguments)
                return {"type": "function_call", "name": tool.function.name, "args": args}
            return {"type": "text", "text": msg.content or ""}
        except Exception as e:
            return {"type": "error", "error": str(e)}


# ------------------------------------------------------------
#  Gemini (structured function calling)
# ------------------------------------------------------------
class GeminiGenerator(BaseGenerator):
    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        if not genai:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt: str, functions: Optional[List[dict]] = None) -> Dict[str, Any]:
        schema = build_function_schemas(functions or [])
        try:
            response = self.model.generate_content([
                {
                    "role": "system",
                    "parts": [
                        f"You are a function-calling model. Available functions: {json.dumps(schema)}",
                        "Respond in JSON: {\"type\":\"function_call\",...}"
                    ]
                },
                {"role": "user", "parts": [prompt]}
            ])
            text = response.text.strip()
            try:
                result = json.loads(text)
                return result
            except Exception:
                return {"type": "text", "text": text}
        except Exception as e:
            return {"type": "error", "error": str(e)}


# ------------------------------------------------------------
#  Fake (for testing)
# ------------------------------------------------------------
class FakeLLM(BaseGenerator):
    def generate(self, prompt: str, functions: Optional[List[dict]] = None) -> Dict[str, Any]:
        if "add" in prompt.lower():
            return {"type": "function_call", "name": "add", "args": {"a": 2, "b": 3}}
        return {"type": "text", "text": "Offline demo: no API used."}


# ------------------------------------------------------------
#  Unified Entry Point
# ------------------------------------------------------------
class LLMClient:
    """
    Unified interface to call different LLMs with function calling support.
    """
    def __init__(self, provider: str = "ollama", **kwargs):
        provider = provider.lower()
        if provider == "ollama":
            self.backend = OllamaGenerator(**kwargs)
        elif provider == "openai":
            self.backend = OpenAIGenerator(**kwargs)
        elif provider == "gemini":
            self.backend = GeminiGenerator(**kwargs)
        elif provider == "fake":
            self.backend = FakeLLM()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def generate(self, prompt: str, functions: Optional[List[dict]] = None) -> Dict[str, Any]:
        return self.backend.generate(prompt, functions)
