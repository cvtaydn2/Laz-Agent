"""Smoke test using httpx."""
from __future__ import annotations
import json, time, httpx

URL = "http://127.0.0.1:8000/v1/chat/completions"
WS  = "C:/Users/Cevat/Documents/Github/Laz-Agent/editor-agent"

TESTS = [
    ("greeting_tr",  "selam",                                 True,  10),
    ("small_talk",   "ordamısın",                             True,  10),
    ("project_q",    "projedeki dosya yapısını kısaca anlat", False, 180),
    ("code_q",       "orchestrator.py ne işe yarıyor?",       False, 180),
    ("bug_q",        "projede bilinen sorunlar var mı?",      False, 180),
]

def chat(msg: str, timeout: int) -> tuple[str, float]:
    payload = {
        "model": "laz-agent",
        "messages": [{"role": "user", "content": msg}],
        "stream": True,
        "extra_body": {"workspace": WS, "mode": "ask"},
    }
    parts: list[str] = []
    t0 = time.monotonic()
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", URL, json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        parts.append(delta)
                except Exception:
                    pass
    return "".join(parts), time.monotonic() - t0

def main():
    print(f"\n{'='*65}")
    print(f"  Laz-Agent Smoke Test  —  model: moonshotai/kimi-k2-instruct")
    print(f"{'='*65}\n")
    ok = fail = 0
    for label, msg, fast, timeout in TESTS:
        print(f"[{label}]  Q: {msg!r}")
        try:
            ans, elapsed = chat(msg, timeout)
            if not ans.strip():
                print(f"  ✗ FAIL — empty response ({elapsed:.2f}s)\n")
                fail += 1
            else:
                tag = " ⚡ fast-path" if fast and elapsed < 1.0 else ""
                print(f"  ✓ OK ({elapsed:.2f}s){tag}")
                print(f"  A: {ans.strip()[:200].replace(chr(10),' ')!r}\n")
                ok += 1
        except Exception as e:
            print(f"  ✗ ERROR — {e}\n")
            fail += 1
    print(f"{'='*65}")
    print(f"  {ok} passed  |  {fail} failed")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()
