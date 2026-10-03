from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic

client = Anthropic()

# stop_reason == "max_token"
def max_token() -> None:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[
            {
                "role": "user",
                "content": "Explain quantum physics."
            }
        ]
    )

    if response.stop_reason == "max_tokens":
        print("You hit the max token limit!")

max_token()
