# microlite/registry.py
from typing import Callable, Dict, Any
import inspect
from dataclasses import dataclass
import json


@dataclass
class FunctionSpec:
    fn: Callable
    name: str
    description: str
    signature: str


class FunctionRegistry:
    def __init__(self):
        self._store: Dict[str, FunctionSpec] = {}

    def register(self, name: str = None, description: str = ""):
        def _decor(fn):
            n = name or fn.__name__
            sig = str(inspect.signature(fn))
            self._store[n] = FunctionSpec(fn=fn, name=n, description=description, signature=sig)
            return fn

        return _decor

    def list_specs(self):
        # return simplified metadata suitable for prompts
        out = []
        for spec in self._store.values():
            out.append({
                "name": spec.name,
                "description": spec.description,
                "signature": spec.signature,
            })
        return out

    def call(self, name: str, argstr_or_args: Any = ""):
        """
        Call a registered function.

        Accepts:
        - dict: call as keyword args
        - list/tuple: call as positional args
        - JSON string: parsed and dispatched
        - k=v pairs string: legacy support
        - empty string: call with no args
        """
        spec = self._store.get(name)
        if not spec:
            raise KeyError(f"Function {name} not found")

        # If caller passed a mapping (common with LLM structured args), use it directly
        if isinstance(argstr_or_args, dict):
            return spec.fn(**argstr_or_args)

        # If caller passed a list/tuple, call positionally
        if isinstance(argstr_or_args, (list, tuple)):
            return spec.fn(*argstr_or_args)

        # Strings: try JSON, then k=v parsing, then single positional
        if isinstance(argstr_or_args, str):
            s = argstr_or_args.strip()
            if not s:
                return spec.fn()

            # attempt JSON parsing
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return spec.fn(**parsed)
                if isinstance(parsed, list):
                    return spec.fn(*parsed)
            except Exception:
                pass

            # legacy k=v parsing
            if "=" in s:
                kwargs = {}
                for kv in s.split():
                    if "=" not in kv:
                        continue
                    k, v = kv.split("=", 1)
                    kwargs[k] = _coerce(v)
                return spec.fn(**kwargs)

            # single positional string
            return spec.fn(s)


def _coerce(s: str):
    # simple coercion for numbers/booleans
    if isinstance(s, (int, float)):
        return s
    if s.isdigit():
        return int(s)
    try:
        f = float(s)
        return f
    except Exception:
        pass
    if isinstance(s, str) and s.lower() in ("true", "false"):
        return s.lower() == "true"
    return s
