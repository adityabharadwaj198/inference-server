Inference server (Qwen/Qwen3-4B)
================================

Loads Qwen/Qwen3-4B with Hugging Face Transformers and runs on CUDA if
available, otherwise CPU. Current entry point: server.ipynb
(server.py is the same flow; it was interrupted with Ctrl+C during the
first weight download).

Setup
-----
From the repo root:

  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r inference-server/requirements.txt

Then open inference-server/server.ipynb with the .venv kernel and run
cells in order:

  0) imports + HF_HOME (inference-server/hf_cache)
  1) device check (so far: CPU, "No GPU available")
  2) load tokenizer + AutoModelForCausalLM (Qwen/Qwen3-4B)
  3) prepare_inputs()

Weights are cached in the default location, ~/.cache/huggingface/hub.
The HF_HOME override was removed (see issue 6).

Issues seen on the terminal, and how they were resolved
-------------------------------------------------------
1) PermissionError on ~/.cache/huggingface/hub while downloading Qwen/Qwen3-4B

   Cause: cache directory owned by another user / leftover lock after a
   canceled download.

   Fix used: sudo chown -R $(id -un):$(id -gn) /Users/adityabharadwaj/.cache
   Follow-up in code: set os.environ["HF_HOME"] to inference-server/hf_cache
   so later runs stay inside the project.

2) ValueError: device_map requires accelerate

   Cause: AutoModelForCausalLM.from_pretrained(..., device_map="auto")
   needs the accelerate package.

   Fix: pip install accelerate  (installed accelerate 1.14.0, exit 0)
   A pip line that also named "library" pulled an unrelated PyPI package
   (library 3.2.4 plus extras like yt-dlp, gallery-dl, praw). That is not
   required for this server and is omitted from requirements.txt.

3) transformers warning: torch_dtype is deprecated

   Newer transformers wants dtype= instead. Renamed in both server.ipynb
   and server.py, so the warning is gone.

4) Unauthenticated Hugging Face Hub requests

   Downloads work without a token but are slower and rate-limited.
   Set HF_TOKEN in the environment for higher limits.

5) KeyboardInterrupt / exit code 1 while downloading model shards

   python3 server.py was stopped with Ctrl+C during snapshot_download
   ("Fetching 3 files", about 331MB at ~2.5MB/s). Traceback ends in
   concurrent.futures thread join, then "Exception ignored on threading
   shutdown" — that is cancel noise, not a model bug.

   Note: an interrupted download does NOT resume. Each run opens a new
   .incomplete blob with a fresh suffix, so the earlier bytes are wasted
   and left behind in the blobs directory. Avoid cancelling cell 2.
   Optional: export HF_TOKEN=... so Hub downloads are faster.

6) HF_HOME does not redirect the weights cache

   Cell 0 (and server.py) do `from transformers import ...` before setting
   os.environ["HF_HOME"]. huggingface_hub resolves the hub cache path at
   import time, so HF_HUB_CACHE is already pinned to ~/.cache/huggingface/hub
   by then. Only the Xet chunk cache, which is resolved later at runtime,
   lands in inference-server/hf_cache.

   Fix applied: dropped the HF_HOME line entirely from cell 0 and server.py.
   The weights already live in ~/.cache/huggingface, so using the default
   location avoids a re-download. If you ever do want a project-local cache,
   set HF_HOME before importing transformers, not after.

7) Xet download hangs behind the corporate proxy

   Cell 2 stalled dead at ~985MB with zero throughput. TCP connections to
   AWS Singapore (s3::get_range) stayed ESTABLISHED but idle, no WARN or
   ERROR in hf_cache/xet/logs. Zscaler silently dropping long-lived range
   requests is the likely cause. Xet also cannot resume across restarts.

   SIGINT does not clear it: Xet downloads run in native Rust threads that
   ignore Python signal handling, so interrupting the cell leaves the kernel
   wedged and still holding the blob locks in
   ~/.cache/huggingface/hub/.locks/models--Qwen--Qwen3-4B. Anything else
   downloading the same repo blocks behind those locks. Killing the kernel
   process is the only way out.

   Fix that worked: disable Xet and fetch from the terminal, which uses
   plain HTTP range requests and resumes automatically.

     HF_HUB_DISABLE_XET=1 .venv/bin/hf download Qwen/Qwen3-4B

   This still hit one truncated response ("peer closed connection without
   sending complete message body") but resumed on its own and completed.

Current runtime notes
---------------------
- Device: CPU ("No GPU available").
- Model: Qwen/Qwen3-4B, 4.02B params, 3 safetensors shards.
- Weights are cached at ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B
  (7.5GB, snapshot 1cfa9a7208912126459214e8b04321603b3df60c).
- Verified: loads from cache in about 1.6s and the chat template renders.
  Cell 2 no longer downloads anything.
- Prefer the terminal `hf download` for cold fetches. The notebook progress
  bar is emitted as an ipywidget but the renderer falls back to text/plain,
  so it looks frozen even when the transfer is healthy.
