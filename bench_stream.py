"""Benchmarks for the streaming post. Needs the server running on :8000.

  A) /generate (wait for everything) vs /stream (first frame) at 32/128/512 tokens
  B) inter-chunk gap distribution on the stream
  C) N concurrent streams
Results are printed and dumped to bench_stream.json.
"""
import json
import statistics as st
import sys
import threading
import time

import httpx

URL = "http://127.0.0.1:8000"
PROMPT = "Count from 1 to 1000 separated by commas. Output only the numbers."
RUNS = 3


def payload(n):
    return {"prompt": PROMPT, "token_limit": n, "temperature": 0.0, "top_k": 1}


def run_generate(n):
    t0 = time.perf_counter()
    r = httpx.post(f"{URL}/generate", json=payload(n), timeout=None)
    r.raise_for_status()
    return {"total_s": time.perf_counter() - t0, "text": r.json()["text"]}


def run_stream(n):
    t0 = time.perf_counter()
    stamps, chunks = [], []
    with httpx.stream("POST", f"{URL}/stream", json=payload(n), timeout=None) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            d = line.removeprefix("data:").lstrip()
            if d == "[DONE]":
                break
            stamps.append(time.perf_counter() - t0)
            chunks.append(json.loads(d)["text"])
    return {
        "ttfb_s": stamps[0],
        "total_s": time.perf_counter() - t0,
        "frames": len(stamps),
        "gaps_ms": [(b - a) * 1000 for a, b in zip(stamps, stamps[1:])],
        "text": "".join(chunks),
    }


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


out = {"A": {}, "C": {}}

print("warmup...", flush=True)
run_stream(8)

print("\n== A/B: generate vs stream ==", flush=True)
for n in (32, 128, 512):
    gens = [run_generate(n) for _ in range(RUNS)]
    strs = [run_stream(n) for _ in range(RUNS)]
    gaps = [g for s in strs for g in s["gaps_ms"]]
    row = {
        "generate_total_s": st.median(g["total_s"] for g in gens),
        "stream_ttfb_s": st.median(s["ttfb_s"] for s in strs),
        "stream_total_s": st.median(s["total_s"] for s in strs),
        "frames": strs[0]["frames"],
        "gap_p50_ms": pct(gaps, 50),
        "gap_p95_ms": pct(gaps, 95),
        "gap_max_ms": max(gaps),
        "same_text": gens[0]["text"] == strs[0]["text"],
        "generate_runs": [g["total_s"] for g in gens],
        "stream_runs": [s["total_s"] for s in strs],
    }
    out["A"][n] = row
    print(n, json.dumps(row), flush=True)

print("\n== C: concurrent streams (64 tokens) ==", flush=True)
for c in (1, 2, 4):
    res = [None] * c

    def work(i):
        res[i] = run_stream(64)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(c)]
    t0 = time.perf_counter()
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.perf_counter() - t0
    row = {
        "ttfb_s_median": st.median(r["ttfb_s"] for r in res),
        "per_stream_total_s": st.median(r["total_s"] for r in res),
        "per_stream_tok_s": st.median(r["frames"] / r["total_s"] for r in res),
        "aggregate_tok_s": sum(r["frames"] for r in res) / wall,
        "wall_s": wall,
    }
    out["C"][c] = row
    print(c, json.dumps(row), flush=True)

json.dump(out, open("bench_stream.json", "w"), indent=2)
