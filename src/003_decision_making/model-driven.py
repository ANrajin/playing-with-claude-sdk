import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic

client = Anthropic()

MODEL = "claude-haiku-4-5-20251001"
SYSTEM_PROMPT = "You are a helpful customer support assistant for an online store."
MAX_TOOL_ITERATIONS = 25

DATA_FILE = Path(__file__).with_name("data.json")
with DATA_FILE.open(encoding="utf-8") as data_file:
    DATA = json.load(data_file)

# ---------------------------------------------------------------------------
# FIX 1: Tool definitions with detailed descriptions.
# Each description answers: what it returns, when to use it (and when to use
# a different tool instead), the input format, and the edge cases.
# ---------------------------------------------------------------------------
our_tools = [
    {
        "name": "get_order",
        "description": (
            "Look up ONE customer order by its numeric order ID (e.g. 1001). "
            "Returns the order status (e.g. 'Pending', 'Shipped'), the customer name, "
            "and 'items': a list of product IDs in the order. "
            "The order does NOT include product names, prices, or stock; call "
            "get_product_details with each product ID to get those. "
            "Use only when the customer gives an order number. "
            "If the order does not exist, the result contains an 'error' field."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "integer",
                    "description": "Numeric order ID, digits only (e.g. 1001). Strip any '#'.",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "get_product_details",
        "description": (
            "Get full details for ONE product by its numeric product ID: name, category, "
            "price, color (if any), and stock. Stock is either a single number (total units) "
            "or, for apparel, an object of units per size (e.g. {\"S\": 4, \"M\": 0}). "
            "0 means out of stock. "
            "Use when you already have a product ID (from get_order or search_products). "
            "If you only know the product NAME, call search_products first to find its ID. "
            "If the product does not exist, the result contains an 'error' field."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "integer",
                    "description": "Numeric product ID (e.g. 3), as returned by get_order or search_products.",
                }
            },
            "required": ["product_id"],
        },
    },
    {
        "name": "search_products",
        "description": (
            "Search the store catalog by keywords. Returns every product whose name, "
            "category, or color contains ALL the query words (case-insensitive), "
            "e.g. 'hoodie', 'blue hoodie', 'laptop', 'keyboard'. "
            "Each result includes full details: id, name, category, price, color, and stock "
            "(per size for apparel), so no follow-up get_product_details call is needed. "
            "Use to find products by name, find alternatives or similar items, "
            "check availability across products, or compare prices. "
            "An empty list means nothing matched; try fewer or broader words "
            "(e.g. 'hoodie' instead of 'blue classic hoodie') before concluding "
            "the store doesn't sell it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to search for, e.g. 'hoodie' or 'laptop'.",
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Optional exact category filter. "
                        "One of: 'Apparel', 'Laptops', 'Accessories'."
                    ),
                },
            },
            "required": ["query"],
        },
    },
]


# Define tools
def get_order(order_id: int) -> dict[str, Any]:
    for order in DATA["orders"]:
        if order["id"] == order_id:
            return order
    return {"error": f"No order found with id {order_id}"}


def get_product_details(product_id: int) -> dict[str, Any]:
    for product in DATA["products"]:
        if product["id"] == product_id:
            return product
    return {"error": f"No product found with id {product_id}"}


# ---------------------------------------------------------------------------
# FIX 2: Word-based search.
# Before: the WHOLE query had to appear as one exact substring, so
# "blue classic hoodie" failed against "classic hoodie apparel blue".
# Now: the query is split into words, and EVERY word must appear somewhere.
# ---------------------------------------------------------------------------
def search_products(query: str, category: str | None = None) -> list[dict[str, Any]]:
    query_words = query.casefold().split()  # "Blue Classic Hoodie" -> ["blue", "classic", "hoodie"]
    normalized_category = category.casefold() if category is not None else None
    matches = []

    for product in DATA["products"]:
        if normalized_category is not None and product["category"].casefold() != normalized_category:
            continue

        searchable_text = " ".join(
            str(product.get(field, "")) for field in ("name", "category", "color")
        ).casefold()

        # all(...) is like LINQ .All(): every query word must be found in the text
        if all(word in searchable_text for word in query_words):
            matches.append(product)

    return matches


# ---------------------------------------------------------------------------
# FIX 3: Errors go back to Claude instead of crashing the agent.
# execute_tool now returns TWO values: (result, is_error).
# - Unknown tool name        -> error result, is_error=True
# - Missing / wrong input    -> error result, is_error=True
# Claude sees the error in the tool_result and can correct itself
# (e.g. retry with the right argument) instead of the program dying.
# ---------------------------------------------------------------------------
TOOL_FUNCTIONS = {
    "get_order": get_order,
    "get_product_details": get_product_details,
    "search_products": search_products,
}


def execute_tool(name: str, tool_input: dict[str, Any]) -> tuple[Any, bool]:
    func = TOOL_FUNCTIONS.get(name)  # like Dictionary.TryGetValue in C#
    if func is None:
        return {
            "error": f"Unknown tool '{name}'.",
            "available_tools": list(TOOL_FUNCTIONS),
        }, True

    try:
        # func(**tool_input) unpacks the dict into named arguments:
        # {"order_id": 1001} -> get_order(order_id=1001)
        return func(**tool_input), False
    except (TypeError, ValueError, KeyError) as exc:
        # TypeError: missing or unexpected argument names
        return {"error": f"Invalid input for tool '{name}': {exc}"}, True


def chat(messages: list[dict[str, Any]]) -> Any:
    for iteration in range(1, MAX_TOOL_ITERATIONS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=our_tools,
            messages=messages,
        )
        print(f"Iteration {iteration}: stop_reason={response.stop_reason}")

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    print(block.text)
            return response

        if response.stop_reason != "tool_use":
            raise RuntimeError(
                f"Unexpected stop_reason {response.stop_reason!r} "
                f"on iteration {iteration}."
            )

        tool_blocks = [block for block in response.content if block.type == "tool_use"]
        for block in tool_blocks:
            print(
                f"Iteration {iteration}: tool={block.name}, "
                f"input={json.dumps(block.input, ensure_ascii=False, sort_keys=True)}"
            )

        if not tool_blocks:
            raise RuntimeError(
                f"Claude returned stop_reason='tool_use' without any tool_use blocks "
                f"on iteration {iteration}."
            )

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in tool_blocks:
            result, is_error = execute_tool(block.name, block.input)  # FIX 3
            if is_error:
                print(f"Iteration {iteration}: tool error -> {result['error']}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False),
                    "is_error": is_error,  # FIX 3: tells Claude this call failed
                }
            )

        messages.append({"role": "user", "content": tool_results})
        if iteration == MAX_TOOL_ITERATIONS:
            raise RuntimeError(
                f"Exceeded the maximum of {MAX_TOOL_ITERATIONS} tool-use iterations."
            )

    raise RuntimeError(f"Exceeded the maximum of {MAX_TOOL_ITERATIONS} iterations.")


def main() -> None:
    question = input("You: ").strip()
    messages = [{"role": "user", "content": question}]
    chat(messages)


if __name__ == "__main__":
    main()