
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extract.hybrid import HybridExtractor  # noqa: E402
from extract.llm import LLMExtractor, make_backend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", choices=["rules", "llm", "hybrid"], default="hybrid")
    ap.add_argument("--backend", choices=["llamacpp", "ollama"], default="llamacpp")
    ap.add_argument("--model", help="GGUF path (llamacpp) or model tag (ollama)")
    ap.add_argument("--input", default=str(ROOT / "eval" / "data" / "gold.jsonl"))
    ap.add_argument("--condition", help="with ASR input: only this condition")
    ap.add_argument("--tag", help="output name (default from extractor+model)")
    ap.add_argument("--no-controlled-confirm", action="store_true")
    ap.add_argument("--n-ctx", type=int, default=4096, help="llamacpp context window")
    a = ap.parse_args()

    llm = None
    if a.extractor in ("llm", "hybrid"):
        if not a.model:
            ap.error("--model is required for llm/hybrid")
        llm = LLMExtractor(make_backend(a.backend, a.model, n_ctx=a.n_ctx))

    rows = [json.loads(l) for l in open(a.input, encoding="utf-8") if l.strip()]
    if a.condition:
        rows = [r for r in rows if r.get("condition") == a.condition]
    is_asr = "raw_text" in rows[0] if rows else False

    tag = a.tag or (a.extractor if not a.model else f"{a.extractor}_{Path(a.model).stem.replace(':', '-')}")
    if a.condition:
        tag += f"_{a.condition}"
    out = ROOT / "eval" / "preds" / f"ext_{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    hybrid = HybridExtractor(llm=llm if a.extractor == "hybrid" else None,
                             confirm_controlled=not a.no_controlled_confirm)
    lat, fails = [], 0
    with open(out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            tl = datetime.fromisoformat(r["time_logged"])
            if a.extractor == "llm":
                try:
                    events, held, meta = llm.extract(r["text"], tl)
                except Exception as exc:  # noqa: BLE001
                    events, held, meta = [], [], {"llm_error": str(exc), "llm_latency_s": 0}
                    fails += 1
                meta["latency_s"] = meta.get("llm_latency_s", 0)
            else:
                events, meta = hybrid.extract(r["text"], tl, asr=r if is_asr else None)
                fails += meta.get("source") == "rules-fallback"
            lat.append(meta.get("latency_s", 0))
            f.write(json.dumps({"id": r["id"], "events": events, "meta": {k: v for k, v in meta.items()
                                                                           if k != "llm_raw"}},
                               ensure_ascii=False, default=str) + "\n")
            flag = sum(e.get("needs_confirmation", False) for e in events)
            print(f"[{i:3}/{len(rows)}] {r['id']} {meta.get('latency_s', 0):6.2f}s  {len(events)} ev  {flag} confirm  {r['text'][:60]}")

    lat_sorted = sorted(lat)
    p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))] if lat_sorted else 0
    print(f"\n{len(rows)} utterances -> {out}")
    print(f"latency per utterance: median {statistics.median(lat):.2f}s  p95 {p95:.2f}s  max {max(lat):.2f}s"
          + (f"   LLM failures: {fails}" if llm else ""))


if __name__ == "__main__":
    main() 