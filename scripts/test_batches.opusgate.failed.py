import time

from dotenv import load_dotenv
import os
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

load_dotenv()

client = Anthropic(
    api_key=os.getenv("OPUSKEY") or os.getenv("ANTHROPIC_API_KEY"),
    base_url="https://api.opusgate.dev",
)

questions = [
    "Say hello in one sentence.",
    "Name one moon of Jupiter.",
    "What is 2 + 2?",
]

batch = client.messages.batches.create(
    requests=[
        Request(
            custom_id=f"q-{i}",
            params=MessageCreateParamsNonStreaming(
                model="claude-opus-5",
                max_tokens=100,
                messages=[{"role": "user", "content": q}],
            ),
        )
        for i, q in enumerate(questions)
    ]
)
print(f"Created batch: {batch.id} (status: {batch.processing_status})")

while True:
    batch = client.messages.batches.retrieve(batch.id)
    print(f"Status: {batch.processing_status}, counts: {batch.request_counts}")
    if batch.processing_status == "ended":
        break
    time.sleep(10)

for result in client.messages.batches.results(batch.id):
    if result.result.type == "succeeded":
        msg = result.result.message
        text = next((b.text for b in msg.content if b.type == "text"), "")
        print(f"[{result.custom_id}] {text}")
    else:
        print(f"[{result.custom_id}] {result.result.type}")
