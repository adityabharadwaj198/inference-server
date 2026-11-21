# tinyinference

A small LLM inference server built from scratch on top of Hugging Face Transformers, to understand what serving frameworks do under the hood. It serves [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) through FastAPI.

- **KV-cached decoding**: one prefill pass, then one token per step.
- **Sampling**: greedy, or temperature plus top-k.
- **SSE streaming**: tokens are sent as they are generated. Streaming decodes the full completion so far and emits only the new suffix, so multi-byte characters such as emoji are never split.
- **Timing metrics**: time to first token (TTFT), latency and throughput.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python server.py          # http://localhost:8000
```

It runs on CUDA if available and falls back to CPU. The first start downloads about 7.5 GB of weights.

## Run it

Start the server in one terminal (wait for the `Loaded Qwen/Qwen3-4B` line):

```bash
source .venv/bin/activate
python server.py
```

Then, in a second terminal, run the streaming test client. It sends a prompt to `/stream` and prints tokens as they arrive:

```bash
source .venv/bin/activate
python server-test.py
```

To change the prompt, token limit, temperature or top-k, edit the `payload` at the top of `server-test.py`.

## API

```bash
curl localhost:8000/health

curl -X POST localhost:8000/generate -H 'content-type: application/json' \
  -d '{"prompt": "What is a KV cache?", "token_limit": 64, "temperature": 0.7, "top_k": 20}'

curl -N -X POST localhost:8000/stream -H 'content-type: application/json' \
  -d '{"prompt": "Write a haiku about GPUs.", "token_limit": 64}'
```

| Field         | Default | Description                              |
|---------------|---------|------------------------------------------|
| `prompt`      | required | User message                            |
| `token_limit` | 32      | Max new tokens (1-4096)                  |
| `temperature` | 0.0     | `0` means greedy decoding                |
| `top_k`       | none    | Sample only from the k most likely tokens |

`/generate` returns `text`, `timing` (`latency_ms`, `ttft_ms`), `input_tokens`, `token_count`, `throughput` (tokens/s) and `cache_usage`. `/stream` emits `data: {"text": "..."}` frames followed by `data: [DONE]`.

## Layout

```
server.py            the server
server-test.py       streaming test client (needs the server running)
scripts/             benchmarks (need the server running)
results/             benchmark output (CPU, Apple laptop)
notebooks/server.ipynb   step-by-step walkthrough that builds the server
```

## Benchmarks

Run from this directory: `python scripts/bench_stream.py` and `python scripts/bench_decode.py`.

On CPU, streaming and `/generate` finish in about the same total time, but streaming shows the first token after about 0.25 s instead of after the whole completion (30 s or more at 512 tokens). Aggregate throughput stays at about 14 tokens/s with 1, 2 or 4 concurrent streams. Requests are serialized, with no batching, so per-stream speed drops as concurrency rises.

Re-decoding the whole completion each step costs 87 µs per step at 512 tokens, and 336 µs at 2000 tokens. That is small next to the 50 ms per-token model step. See `results/`.

## Limitations

- No continuous batching, so concurrent requests share the model one at a time.
- No stop sequences, repetition penalty or top-p.
- Streaming re-decodes the whole completion each step, which is O(n²) in total. It is negligible at these lengths.

## Troubleshooting downloads behind a proxy

If the Hub download hangs, for example behind Zscaler, disable Xet and download from the terminal. This uses plain HTTP range requests and resumes automatically:

```bash
HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen3-4B
```

Set `HF_TOKEN` for higher Hub rate limits.
