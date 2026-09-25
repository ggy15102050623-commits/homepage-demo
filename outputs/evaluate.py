from __future__ import annotations

import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent
QUESTIONS = json.loads((OUTPUT / "eval_questions.json").read_text(encoding="utf-8-sig"))
RAW_PATH = ROOT / "work" / "evaluation_raw.json"

results = []
for item in QUESTIONS:
    print(f"[{item['id']}] {item['question']}", flush=True)
    started = time.time()
    try:
        response = requests.post(
            "http://127.0.0.1:8765/api/ask",
            json={"query": item["question"], "top_k": 6},
            timeout=180,
        )
        payload = response.json()
        result = {
            **item,
            "http_status": response.status_code,
            "elapsed_seconds": round(time.time() - started, 2),
            "answer": payload.get("answer", ""),
            "hits": [
                {
                    "id": hit.get("id"),
                    "company": hit.get("company"),
                    "stock_code": hit.get("stock_code"),
                    "chapter": hit.get("chapter"),
                    "page": hit.get("page"),
                    "semantic_score": hit.get("semantic_score"),
                    "bm25_score": hit.get("bm25_score"),
                    "excerpt": hit.get("excerpt", ""),
                }
                for hit in payload.get("hits", [])
            ],
            "error": payload.get("error", ""),
        }
    except Exception as exc:
        result = {**item, "http_status": 0, "elapsed_seconds": round(time.time() - started, 2), "answer": "", "hits": [], "error": type(exc).__name__ + ": " + str(exc)}
    results.append(result)
    print(f"  HTTP {result['http_status']} | {result['elapsed_seconds']}s | hits={','.join(h['id'] for h in result['hits'])}", flush=True)
    if result.get("error"):
        print(f"  ERROR: {result['error']}", flush=True)
    elif result.get("answer"):
        print("  " + result["answer"].replace("\n", " ")[:420], flush=True)
    time.sleep(0.2)

RAW_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Saved: {RAW_PATH}")
