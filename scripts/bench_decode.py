"""Cost of re-decoding the whole completion every step (tokenizer only, no model).
Compares against decoding just the new token, which is the broken-for-emoji shortcut."""
import json
import statistics as st
import time

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
ids = tok(" ".join(str(i) for i in range(1, 3000)), add_special_tokens=False)["input_ids"]
print(tok.decode(ids, skip_special_tokens=True))
rows = {}
for n in (32, 128, 512, 2000):
    seq = ids[:n]
    full, last = [], []
    for _ in range(200):
        t = time.perf_counter(); tok.decode(seq, skip_special_tokens=True); full.append(time.perf_counter() - t)
        t = time.perf_counter(); tok.decode(seq[-1:], skip_special_tokens=True); last.append(time.perf_counter() - t)
    # total decode time across a whole n-token generation (step k decodes k tokens)
    t = time.perf_counter()
    for k in range(1, n + 1):
        tok.decode(ids[:k], skip_special_tokens=True)
    cumulative = time.perf_counter() - t
    rows[n] = {
        "full_decode_us": st.median(full) * 1e6,
        "last_only_us": st.median(last) * 1e6,
        "cumulative_ms": cumulative * 1000,
    }
    print(n, json.dumps(rows[n]), flush=True)
json.dump(rows, open("results/bench_decode.json", "w"), indent=2)
