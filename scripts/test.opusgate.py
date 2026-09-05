import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url="https://api.opusgate.dev",
)

response = client.messages.create(
    model="claude-opus-5",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "Say hello in one sentence, tell me what your model exact version is."}
    ],
)

print(response.content[0].text)