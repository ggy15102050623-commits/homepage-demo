from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def last_token_pool(hidden: torch.Tensor) -> torch.Tensor:
    # The Qwen embedding tokenizer uses left padding, so the final position is the last real token.
    return hidden[:, -1, :]


def main() -> None:
    default_input = Path(__file__).resolve().parent.parent / "work" / "knowledge_base_2026.jsonl"
    if not default_input.is_file():
        default_input = Path(r"D:\AI任务\知识库\knowledge_base_2026.jsonl")
    default_model = Path(r"D:\AI任务\知识库\models\Qwen3-Embedding-0.6B")
    parser = argparse.ArgumentParser(description="Generate a local Qwen3 embedding index from half-year reports.")
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "work" / "vector_index.npz")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"JSONL knowledge base not found: {args.input}")
    if not args.model.is_dir():
        raise FileNotFoundError(f"Local embedding model not found: {args.model}")
    records = [json.loads(line) for line in args.input.open("r", encoding="utf-8") if line.strip()]
    if not records:
        raise ValueError("The input JSONL contains no records.")

    company_counts = collections.Counter(
        record.get("metadata", {}).get("company", "未知公司") for record in records
    )
    print(f"Input chunks: {len(records)} | companies: {len(company_counts)}")
    print(f"Model: {args.model}")
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32
    if use_cuda:
        print(f"Device: {torch.cuda.get_device_name(0)} | dtype: float16")
    else:
        print("Device: CPU | dtype: float32")

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), padding_side="left", local_files_only=True
    )
    model = AutoModel.from_pretrained(
        str(args.model),
        dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device)
    model.eval()
    print(f"Model loaded in {time.time() - start:.1f}s")

    contents = [str(record.get("content", "")) for record in records]
    vectors = np.empty((len(contents), 1024), dtype=np.float16 if use_cuda else np.float32)
    batch_size = max(1, args.batch_size)
    cursor = 0
    last_report = time.time()
    while cursor < len(contents):
        current = min(batch_size, len(contents) - cursor)
        batch = contents[cursor : cursor + current]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        try:
            with torch.inference_mode():
                output = model(**tokens, use_cache=False)
                embeddings = F.normalize(last_token_pool(output.last_hidden_state).float(), p=2, dim=1)
            vectors[cursor : cursor + current] = embeddings.cpu().numpy().astype(vectors.dtype, copy=False)
            cursor += current
        except torch.cuda.OutOfMemoryError:
            if not use_cuda or batch_size <= 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            print(f"GPU memory pressure; retrying with batch size {batch_size}")
            continue
        if cursor == len(contents) or time.time() - last_report >= 10:
            elapsed = max(0.1, time.time() - start)
            rate = cursor / elapsed
            remaining = (len(contents) - cursor) / max(rate, 0.001)
            print(f"Embedded {cursor}/{len(contents)} chunks ({rate:.1f}/s; ETA {remaining/60:.1f} min)")
            last_report = time.time()

    print("Projecting the 1024-dimensional vectors into a 2D map...")
    coordinates = PCA(n_components=2, svd_solver="randomized", random_state=42).fit_transform(
        vectors.astype(np.float32)
    )
    for axis in range(2):
        low = float(coordinates[:, axis].min())
        high = float(coordinates[:, axis].max())
        span = max(high - low, 1e-9)
        coordinates[:, axis] = 0.035 + 0.93 * (coordinates[:, axis] - low) / span

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        embeddings=vectors,
        coordinates=coordinates.astype(np.float32),
    )
    temporary.replace(args.output)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": str(args.input),
        "input_sha256": sha256_file(args.input),
        "model": "Qwen3-Embedding-0.6B",
        "model_path": str(args.model),
        "device": torch.cuda.get_device_name(0) if use_cuda else "CPU",
        "dtype": str(vectors.dtype),
        "dimensions": int(vectors.shape[1]),
        "chunks": int(vectors.shape[0]),
        "companies": dict(sorted(company_counts.items())),
        "index_file": str(args.output),
        "index_size_bytes": args.output.stat().st_size,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    manifest_path = args.output.with_name("index_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved index: {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MiB)")
    print(f"Saved manifest: {manifest_path}")
    print(f"Completed in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
