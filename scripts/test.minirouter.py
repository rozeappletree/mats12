import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["MINIROUTER_KEY"],
    base_url="https://api.minirouter.sh/v1",
)

stream = client.chat.completions.create(
    model="alibaba/qwen3.7-plus",
    # model="meta/muse-spark-1.2",
    # model="zai/glm-5.3-flash",
    # model="spacexai/grok-4.6",
    messages=[{"role": "user", "content": "fav animal?"}],
    stream=True,
)

chunks = []
for event in stream:
    delta = event.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
        chunks.append(delta)
print()

print("--- full response ---")
print("".join(chunks))
