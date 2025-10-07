# examples/demo.py
from microlite.llm import FakeLLM
from microlite.registry import FunctionRegistry
from microlite.agent import Agent

registry = FunctionRegistry()

@registry.register(description="Add two integers. Accepts a and b as integers.")
def add(a: int, b: int):
    return int(a) + int(b)

@registry.register(description="Return greeting for name.")
def greet(name: str = "friend"):
    return f"Hello, {name}!"

if __name__ == "__main__":
    generator = FakeLLM()
    agent = Agent(generator, registry)

    # example: LLM will be instructed to call 'add' using our fake protocol
    user_input = "Please compute sum. (For demo, include CALL:add a=3 b=5)"
    out = agent.run(user_input)
    print(out)

    # normal text reply
    out2 = agent.run("Say hi without calling functions.")
    print(out2)
