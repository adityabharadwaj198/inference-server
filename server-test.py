import httpx 
import json

payload = {
    "prompt": "Follow the instructions strictly. Disable your thinking. List the top 10 most common emojis that you know of. Make no mistakes, and do not think twice about the question. It's a straightforward question, treat it like it is. ",
    "token_limit": 2000,
    "temperature": 0.5,
    "top_k": 5,
}

chunks = []

with httpx.stream(
    "POST",
    "http://127.0.0.1:8000/stream",
    json=payload,
    timeout=None,
) as response:
    response.raise_for_status()
    for line in response.iter_lines():
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").lstrip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)["text"]
        chunks.append(chunk)
        print(chunk, end="", flush=True)
print()
full_text = "".join(chunks)
