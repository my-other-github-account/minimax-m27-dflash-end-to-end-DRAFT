"""Example: serve a DFlash drafter via llama.cpp's OpenAI-compatible server.

Run on spark-1::

    python repro/examples/03_serve_openai.py

Then from anywhere::

    curl http://spark-1:8080/v1/chat/completions \\
      -H 'Content-Type: application/json' \\
      -d '{
        "model": "minimax-m2.7-dflash",
        "messages": [{"role": "user", "content": "Write a fibonacci function."}],
        "max_tokens": 256
      }'
"""
from dflash_llama import LlamaServer

VERIFIER = "/home/user/clawd/iq4_models/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf"
DRAFTER  = "/home/user/models/MiniMax-M2.7-DFlash-bf16only.gguf"

# Context-manager form — auto-cleanup on exit
with LlamaServer(
    verifier_gguf=VERIFIER,
    drafter_gguf=DRAFTER,           # remove this for verifier-only AR baseline
    spec_type="dflash",
    draft_max=7,
    host="0.0.0.0",
    port=8080,
    ctx=8192,
    log_path="/tmp/llama-server.log",
) as srv:
    print(f"OpenAI-compatible endpoint: {srv.url}")
    print(f"Try: curl {srv.url}/models")

    # Demo: hit the endpoint with the openai client (optional dep)
    try:
        from openai import OpenAI
        client = OpenAI(base_url=srv.url, api_key="not-needed")
        resp = client.chat.completions.create(
            model="dflash",
            messages=[{"role": "user",
                       "content": "Write a one-line Python fibonacci."}],
            max_tokens=64,
        )
        print("Reply:", resp.choices[0].message.content)
    except ImportError:
        print("(install 'openai' to test inline; otherwise hit the endpoint with curl)")
        # Idle so the server stays up for an external client
        import time
        while True:
            time.sleep(60)
