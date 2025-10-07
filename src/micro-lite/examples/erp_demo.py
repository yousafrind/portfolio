from microlite.llm import BaseGenerator
from microlite.registry import FunctionRegistry
from microlite.agent import Agent
from typing import List, Dict, Any


class DeterministicLLM(BaseGenerator):
    """Simple deterministic LLM simulator that inspects the prompt and
    returns structured function_call responses for demo purposes.
    """

    def generate(self, prompt: str, functions: List[dict] = None) -> Dict[str, Any]:
        # Extract the user's input from the full prompt to avoid matching
        # function names or other template text that may be present.
        p_raw = prompt
        if "user:" in prompt.lower():
            try:
                # split on case-insensitive 'User:' and take the last segment
                parts = prompt.split("User:")
                p_raw = parts[-1]
            except Exception:
                p_raw = prompt

        p = p_raw.strip().splitlines()[0].lower() if p_raw.strip() else prompt.lower()
        # create order intent
        if "create order" in p or "place order" in p or "order" in p:
            # return a sample structured call for create_order
            return {
                "type": "function_call",
                "name": "create_order",
                "args": {
                    "customer_id": "C123",
                    "items": [{"sku": "SKU1", "qty": 2}, {"sku": "SKU2", "qty": 1}]
                }
            }

        # check inventory intent
        if "inventory" in p or "stock" in p or "check" in p:
            return {"type": "function_call", "name": "get_inventory", "args": {"sku": "SKU1"}}

        # list orders
        if "list orders" in p or "orders for" in p:
            return {"type": "function_call", "name": "list_orders", "args": {"customer_id": "C123"}}

        # get customer
        if "customer" in p:
            return {"type": "function_call", "name": "get_customer", "args": {"customer_id": "C123"}}

        return {"type": "text", "text": "I didn't understand that in demo mode."}


### In-memory ERP store and functions

inventory = {"SKU1": 10, "SKU2": 5}
customers = {"C123": {"name": "Acme Corp", "email": "poc@acme.example"}}
orders: List[Dict[str, Any]] = []


def register_erp_functions(registry: FunctionRegistry):
    @registry.register(description="Get customer details by customer_id")
    def get_customer(customer_id: str):
        return customers.get(customer_id, {})

    @registry.register(description="Get inventory level for SKU")
    def get_inventory(sku: str):
        return {"sku": sku, "qty": inventory.get(sku, 0)}

    @registry.register(description="Create an order for a customer. items is a list of {sku, qty}.")
    def create_order(customer_id: str, items: List[Dict[str, Any]]):
        # simple stock check and decrement
        for it in items:
            sku = it["sku"]
            qty = int(it.get("qty", 1))
            if inventory.get(sku, 0) < qty:
                return {"error": f"Not enough stock for {sku}"}

        # decrement
        for it in items:
            sku = it["sku"]
            qty = int(it.get("qty", 1))
            inventory[sku] = inventory.get(sku, 0) - qty

        order_id = f"ORD{len(orders)+1:04d}"
        order = {"order_id": order_id, "customer_id": customer_id, "items": items}
        orders.append(order)
        return {"order_id": order_id, "status": "created"}

    @registry.register(description="List orders for a customer")
    def list_orders(customer_id: str):
        return [o for o in orders if o["customer_id"] == customer_id]

    @registry.register(description="Update inventory for a SKU by delta (can be negative)")
    def update_inventory(sku: str, delta: int):
        inventory[sku] = inventory.get(sku, 0) + int(delta)
        return {"sku": sku, "qty": inventory[sku]}


def run_demo():
    registry = FunctionRegistry()
    register_erp_functions(registry)

    generator = DeterministicLLM()
    agent = Agent(generator, registry)

    # Simulate a short conversation
    turns = [
        "Create an order for Acme Corp for 2 of SKU1 and 1 of SKU2",
        "What's the inventory for SKU1?",
        "Show orders for customer C123",
        "Update inventory for SKU1 by 5",
        "Check inventory for SKU1 again",
    ]

    for t in turns:
        print(f"\nUser: {t}")
        out = agent.run(t)
        print("Agent output:", out)


if __name__ == "__main__":
    run_demo()
