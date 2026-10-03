from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic

client = Anthropic()

# stop_reason == "end_turn"
def end_turn() -> None:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": "Hey, claude!"
            }
        ]
    )

    if response.stop_reason == "end_turn":
        # Process the complete response
        for block in response.content:
            if block.type == "text":
                print(block.text)

end_turn()
