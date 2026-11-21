"""A tiny LLM inference server: Qwen3-4B + KV cache + SSE streaming, behind FastAPI.

Endpoints:
    GET  /health    liveness check
    POST /generate  full completion with timing metrics
    POST /stream    token-by-token Server-Sent Events

Run:  python server.py   (serves on http://0.0.0.0:8000)
"""

import json
import time
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B"
SYSTEM_PROMPT = (
    "You are a good assistant. Keep your responses short and concise. "
    "Do not overthink your responses. Do not make up facts. Do not hallucinate. "
    "Ground your responses in reality."
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

_load_start = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map="auto"
).eval()
print(f"Loaded {MODEL_ID} in {time.time() - _load_start:.2f}s")


def sync_gpu() -> None:
    """Wait for queued CUDA work so wall-clock timings are accurate."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prepare_inputs(prompt: str) -> dict:
    """Render the chat template for `prompt` and move the tensors to the model's device."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    return inputs.to(model.device)


def select_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Pick the next token from `logits` of shape (batch, vocab): greedy, or temperature + top-k sampling."""
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled_logits = logits / temperature
    if top_k is not None:
        cutoff = torch.topk(scaled_logits, top_k, dim=-1).values[:, -1:]
        scaled_logits = scaled_logits.masked_fill(scaled_logits < cutoff, float("-inf"))

    probabilities = scaled_logits.softmax(dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


@torch.inference_mode()
def generate_cached(
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
) -> dict:
    """Generate a full completion, reusing the KV cache between steps."""
    inputs = prepare_inputs(prompt)
    generated_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_length = generated_ids.shape[1]
    first_token_at = None

    sync_gpu()
    started_at = time.perf_counter()

    # Prefill: process the whole prompt once and keep the KV cache.
    outputs = model(input_ids=generated_ids, attention_mask=attention_mask, use_cache=True)
    past_key_values = outputs.past_key_values

    # Decode: feed one token at a time, attending to the cache.
    for _ in range(max_new_tokens):
        next_token = select_next_token(outputs.logits[:, -1, :], temperature, top_k)
        if next_token.item() == tokenizer.eos_token_id:
            break

        if first_token_at is None:
            sync_gpu()
            first_token_at = time.perf_counter()

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values

    sync_gpu()
    finished_at = time.perf_counter()
    completion_ids = generated_ids[0, prompt_length:]
    return {
        "text": tokenizer.decode(completion_ids, skip_special_tokens=True),
        "token_ids": completion_ids.tolist(),
        "input_tokens": prompt_length,
        "output_tokens": completion_ids.shape[0],
        "ttft_ms": ((first_token_at or finished_at) - started_at) * 1000,
        "latency_ms": (finished_at - started_at) * 1000,
    }


@torch.inference_mode()
def generate_stream(
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
):
    """Yield SSE frames as tokens are produced.

    StreamingResponse iterates this generator and flushes each yield to the client.
    """
    if top_k is not None and top_k <= 0:
        top_k = None

    inputs = prepare_inputs(prompt)
    generated_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_length = generated_ids.shape[1]
    previous_text = ""

    outputs = model(input_ids=generated_ids, attention_mask=attention_mask, use_cache=True)
    past_key_values = outputs.past_key_values

    for _ in range(max_new_tokens):
        next_token = select_next_token(outputs.logits[:, -1, :], temperature, top_k)
        if next_token.item() == tokenizer.eos_token_id:
            break

        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

        # Decode the whole completion so far and emit only the new suffix: a single
        # token is not always a complete UTF-8 / BPE piece (e.g. emoji), and a
        # trailing U+FFFD means the character is still incomplete.
        current_text = tokenizer.decode(
            generated_ids[0, prompt_length:], skip_special_tokens=True
        )
        if not current_text.endswith("�"):
            chunk = current_text[len(previous_text):]
            previous_text = current_text
            if chunk:
                yield f"data: {json.dumps({'text': chunk})}\n\n"

        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = outputs.past_key_values

    yield "data: [DONE]\n\n"


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The prompt to generate from")
    token_limit: int = Field(32, ge=1, le=4096, description="Maximum number of tokens to generate")
    temperature: float = Field(0.0, ge=0.0, description="Sampling temperature; 0 means greedy")
    top_k: Optional[int] = Field(None, description="Sample only from the top-k tokens")


class GenerateResponse(BaseModel):
    text: str = Field(..., description="The generated text")
    timing: dict = Field(..., description="latency_ms and ttft_ms")
    input_tokens: int = Field(..., description="Prompt length in tokens")
    token_count: int = Field(..., description="Number of generated tokens")
    throughput: float = Field(..., description="Generated tokens per second")
    cache_usage: str = Field(..., description="Whether the KV cache was used")


app = FastAPI(title="tinyinference")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    result = generate_cached(
        request.prompt, request.token_limit, request.temperature, request.top_k
    )
    seconds = result["latency_ms"] / 1000
    return GenerateResponse(
        text=result["text"],
        timing={"latency_ms": result["latency_ms"], "ttft_ms": result["ttft_ms"]},
        input_tokens=result["input_tokens"],
        token_count=result["output_tokens"],
        throughput=result["output_tokens"] / seconds if seconds else 0.0,
        cache_usage="kv_cache",
    )


@app.post("/stream")
def stream_text(request: GenerateRequest):
    # Plain `def` so Starlette runs the blocking generator in a threadpool
    # instead of stalling the event loop between tokens.
    return StreamingResponse(
        generate_stream(
            request.prompt, request.token_limit, request.temperature, request.top_k
        ),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
