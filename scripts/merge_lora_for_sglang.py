#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter and convert the saved model layout for SGLang.

Qwen3.5-9B in this workspace serves correctly in SGLang when the merged
weights use the base model's multimodal config sidecar files and shard names
like `model.safetensors-00001-of-00005.safetensors`.  A vanilla Transformers
text merge saves `model-00001-of-00005.safetensors` and a text-only config,
which SGLang did not accept in the GRPO step-500 run.  This script performs
both steps deterministically.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_SHARD_RE = re.compile(r"^(model|pytorch_model).*(\.safetensors|\.bin)$")


def copy_non_weight_sidecars(src: Path, dst: Path, *, overwrite: bool = True) -> None:
    for p in src.iterdir():
        if p.is_dir():
            if p.name == ".cache":
                target = dst / p.name
                if target.exists() and overwrite:
                    shutil.rmtree(target)
                if not target.exists():
                    shutil.copytree(p, target)
            continue
        name = p.name
        if name == "model.safetensors.index.json" or MODEL_SHARD_RE.match(name):
            continue
        target = dst / name
        if overwrite or not target.exists():
            shutil.copy2(p, target)


def convert_weight_files(text_dir: Path, sglang_dir: Path) -> dict:
    idx_path = text_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing index after merge: {idx_path}")
    idx = json.loads(idx_path.read_text(encoding="utf-8"))

    mapping = {}
    for fname in sorted(set(idx["weight_map"].values())):
        src = text_dir / fname
        if not src.exists():
            raise FileNotFoundError(src)
        # Transformers commonly writes model-00001-of-00005.safetensors; SGLang
        # in this workspace accepted model.safetensors-00001-of-00005.safetensors.
        new_name = re.sub(r"^model-", "model.safetensors-", fname)
        new_name = re.sub(r"^pytorch_model-", "model.safetensors-", new_name)
        dst = sglang_dir / new_name
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
        mapping[fname] = new_name

    idx["weight_map"] = {k: mapping.get(v, v) for k, v in idx["weight_map"].items()}
    (sglang_dir / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2, sort_keys=True), encoding="utf-8")
    return mapping


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--merged-text", required=True)
    ap.add_argument("--sglang-dir", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-shard-size", default="4GB")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base)
    adapter = Path(args.adapter)
    merged_text = Path(args.merged_text)
    sglang_dir = Path(args.sglang_dir)
    if not adapter.exists():
        raise FileNotFoundError(adapter)
    for out in (merged_text, sglang_dir):
        if out.exists() and args.force:
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"[merge] base={base}", flush=True)
    print(f"[merge] adapter={adapter}", flush=True)
    print(f"[merge] merged_text={merged_text}", flush=True)
    print(f"[merge] sglang_dir={sglang_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    print("[merge] merging adapter into base...", flush=True)
    model = model.merge_and_unload()
    model.save_pretrained(merged_text, safe_serialization=True, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(merged_text)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[convert] copying sidecars and converting shard filenames...", flush=True)
    copy_non_weight_sidecars(base, sglang_dir, overwrite=True)
    copy_non_weight_sidecars(merged_text, sglang_dir, overwrite=True)
    # Keep the base multimodal config that SGLang expects for this Qwen3.5 build.
    for fname in ("config.json", "preprocessor_config.json", "video_preprocessor_config.json", "merges.txt", "vocab.json"):
        src = base / fname
        if src.exists():
            shutil.copy2(src, sglang_dir / fname)
    mapping = convert_weight_files(merged_text, sglang_dir)
    meta = {
        "merged_utc": dt.datetime.now(dt.UTC).isoformat(),
        "base": str(base),
        "adapter": str(adapter),
        "merged_text": str(merged_text),
        "sglang_dir": str(sglang_dir),
        "dtype": args.dtype,
        "max_shard_size": args.max_shard_size,
        "mapping_examples": list(mapping.items())[:3],
    }
    (merged_text / "merge_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (sglang_dir / "merge_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] sglang model ready: {sglang_dir}", flush=True)


if __name__ == "__main__":
    main()
