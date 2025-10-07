# micro-lite

Lightweight micro framework for experimenting with LLM function-calling and building simple agent-based integrations.

This project provides a tiny Python package `microlite` with:

- LLM adapter stubs and a unified client (`microlite/llm.py`)
- A function registry to register Python callables (`microlite/registry.py`)
- An agent that composes prompts, calls an LLM backend, and dispatches function calls (`microlite/agent.py`)
- A prompt template helper (`microlite/templates.py`)
- Small examples demonstrating usage (`examples/demo.py`, `examples/erp_demo.py`)

This repository is intended as a development playground and teaching scaffold for building conversational integrations (for example, a lightweight conversational ERP).

## Features

- Minimal components to experiment with function-calling workflows
- Backwards-compatible registry that accepts structured args (dict/list/JSON) and legacy `k=v` strings
- Pluggable LLM backends (local Ollama, OpenAI/Gemini adapters sketched, plus `FakeLLM` for offline demos)
- Example showing how to wire the pieces together into an ERP-like demo

## Repo layout

```
micro-lite/
├─ microlite/
│  ├─ __init__.py
│  ├─ llm.py
│  ├─ registry.py
│  ├─ agent.py
│  └─ templates.py
├─ examples/
│  ├─ demo.py
│  └─ erp_demo.py
├─ requirements.txt
├─ repo_desc.md
└─ README.md
```

## Quickstart (PowerShell)

1. Install locally in editable mode (recommended) or run with PYTHONPATH.

```powershell
python -m pip install -e .
python .\examples\demo.py
```

Or without installing:

```powershell
$env:PYTHONPATH = "$PWD\micro-lite"
python .\micro-lite\examples\demo.py
```

2. Run the ERP demo (no external APIs required):

```powershell
$env:PYTHONPATH = "$PWD\micro-lite"
python .\micro-lite\examples\erp_demo.py
```

The `demo.py` uses a `FakeLLM` to simulate function-calling. The `erp_demo.py` demonstrates a simple conversational ERP flow (create orders, check/update inventory, list orders) using an in-memory store.

## How it works (conceptually)

- Register Python functions with `FunctionRegistry` using the `@registry.register(...)` decorator. Each registered function is stored with metadata (name, description, signature).
- The `Agent` composes a prompt with available function metadata and the user's input, then calls the configured LLM backend.
- LLMs should return a normalized response shape: a dict with `type` = `function_call` (with `name` and `args`), `text`, or `error`.
- When the Agent receives a `function_call`, it forwards the call to `FunctionRegistry.call(...)`. The registry supports dict args, lists, JSON strings, `k=v` strings, and single positional args.

## Examples

- `examples/demo.py` — minimal demo registering `add` and `greet` functions and running the agent with `FakeLLM`.
- `examples/erp_demo.py` — a more complete demo that registers ERP-like functions (get_customer, get_inventory, create_order, list_orders, update_inventory) and runs a short scripted conversation using a deterministic LLM simulator.

## Extending to a real LLM

To use a real provider:

1. Install the SDK you want (for example `openai`).
2. Update or configure `microlite/llm.py` to use the provider class (`OpenAIGenerator`, `GeminiGenerator`, or custom adapter).
3. Ensure the provider returns structured function call responses (name + args) or adapt the adapter to map provider outputs to the normalized shape used by `Agent`.

Note: `llm.py` currently contains example adapters and helper code; depending on the version of provider SDKs you use, you may need to adjust the call signatures.

## Recommended next work

- Add `microlite/__init__.py` exports for a clearer public API.
- Add unit tests (pytest) for `FunctionRegistry.call` to validate dict/list/JSON/k=v parsing.
- Add session/context state handling for multi-turn conversations (current customer, order-in-progress, etc.).
- Add examples wiring this to a small web UI or chat server.

## Troubleshooting

- If imports fail when running examples, ensure `PYTHONPATH` includes the package root or install in editable mode.
- If using a real LLM provider, check SDK versions and environment variables (API keys).

## License

This project is provided as-is for experimentation. Add a license file (e.g., MIT) if you plan to publish this repository.

---

If you want, I can add a `__init__.py` export, a pytest suite, or wire up the OpenAI adapter next — which would you prefer?
