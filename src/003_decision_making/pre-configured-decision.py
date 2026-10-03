import json
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic


def log_step(message: str) -> None:
    print(f"[step] {message}")


log_step("Initializing the Claude client.")
client = Anthropic()

DATA_FILE = Path(__file__).with_name("data.json")
log_step(f"Loading order and product data from {DATA_FILE.name}.")
with DATA_FILE.open(encoding="utf-8") as data_file:
    DATA = json.load(data_file)

ORDER_NUMBER_PATTERN = re.compile(
    r"\border(?:\s+(?:number|no\.?|id))?\s*(?:#|:)?\s*(\d{4,})\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"\b\d{4,}\b")


def get_order(order_id: int) -> dict:
    for order in DATA["orders"]:
        if order["id"] == order_id:
            return order
    return {"error": f"No order found with id {order_id}"}


def get_product_details(product_id: int) -> dict:
    for product in DATA["products"]:
        if product["id"] == product_id:
            return product
    return {"error": f"No product found with id {product_id}"}


def answer_question(question: str) -> None:
    log_step("Searching the user's message for an order number.")
    match = ORDER_NUMBER_PATTERN.search(question) or NUMBER_PATTERN.search(question)
    prompt = question

    if match:
        order_id = int(match.group(1) if match.lastindex else match.group())
        log_step(f"Found order number {order_id}.")
        log_step(f"Calling get_order({order_id}).")
        order = get_order(order_id)
        gathered_data = {"order": order}

        if "items" in order:
            product_details = []
            for item_number, product_id in enumerate(order["items"], start=1):
                log_step(
                    f"Calling get_product_details({product_id}) "
                    f"for item {item_number} of {len(order['items'])}."
                )
                product_details.append(get_product_details(product_id))
            gathered_data["products"] = product_details

        prompt = (
            f"User question:\n{question}\n\n"
            f"Retrieved order data:\n"
            f"{json.dumps(gathered_data, ensure_ascii=False, indent=2)}\n\n"
            "Write a helpful answer to the user's question using the retrieved data."
        )
        log_step("Prepared the question with the retrieved order and product data.")
    else:
        log_step("No order number found; sending the question without additional data.")

    log_step("Sending one request to Claude with no tools.")
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    log_step("Printing Claude's final reply.")
    for block in response.content:
        if block.type == "text":
            print(block.text)


def main() -> None:
    log_step("Waiting for the user's question.")
    question = input("You: ").strip()
    answer_question(question)


if __name__ == "__main__":
    main()
