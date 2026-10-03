from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic

client = Anthropic()

def stop_sequence() -> None:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        stop_sequences=["END", "STOP"],
        messages=[
            {
                "role": "user",
                "content": "Generate text until you say END"
            }
        ]
    )

    if response.stop_reason == "stop_sequence":
        print(f"Stopped at sequence: {response.stop_sequence}")
        
        for block in response.content:
            if block.type == "text":
                print(block.text)

stop_sequence()
