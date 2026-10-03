from dotenv import load_dotenv

load_dotenv()

from anthropic import Anthropic

client = Anthropic()

# Define the tool
our_tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties":{
                "location": {
                    "type": "string",
                    "description": "City and state"
                }
            },
            "required": ["location"]
        }
    }
]

# Execute the tool
def get_weather(location):
    return f"Weather in {location}: 72°F"

def execute_tool(name, tool_input):
    """Executes a tool and return the result"""
    if name == "get_weather":
        return get_weather(tool_input.get("location", "unknown"))

def chat(messages):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        tools=our_tools,
        messages=messages
    )

    return response

def add_user_message(messages, content):
    messages.append({"role": "user", "content": content})

def add_assistant_message(messages, content):
    messages.append({"role": "assistant", "content": content})

def main() -> None:
    messages = []

    add_user_message(messages, "what is the weather in San Francisco?")
    answer = chat(messages)

    add_assistant_message(messages, answer.content)

    if answer.stop_reason == "tool_use":
        tool_result = []
        for block in answer.content:
            if block.type == "tool_use":
                response = execute_tool(block.name, block.input)
                tool_result.append({"type": "tool_result", "tool_use_id": block.id, "content": response})
                add_user_message(messages, tool_result)

    answer = chat(messages)

    if answer.stop_reason == "end_turn":
        print(answer.content[0].text)

main()
