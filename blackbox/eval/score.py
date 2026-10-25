"""
Black Box - scorer.

Extraction (per-field precision / recall / F1):
  python eval/score.py --pred eval/preds/rules.jsonl --label rules-v1

ASR (WER + safety-critical recall), optionally together with extraction:
  python eval/score.py --asr eval/preds/asr_small.jsonl
  python eval/score.py --pred eval/preds/rules_on_siren5.jsonl --asr eval/preds/asr_small.jsonl --label rules+whisper-small

Every run with --label is appended to eval/results/history.csv, so you can show
the score climbing feature by feature in the pitch.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from normalize import words_to_numbers  # noqa: E402
from protocol import ProtocolBook, to_canonical  # noqa: E402

BOOK = ProtocolBook.load()
VITAL_KEYS = ["sbp", "dbp", "hr", "spo2", "rr", "gcs", "bgl_mg_dl", "temp_c"]
HEADLINE = ["drug", "dose", "unit", "route", "time"]


# ------------------------------------------------------------------ helpers
def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def event_key(ev: dict) -> str:
    t = ev.get("event_type")
    return {"medication": ev.get("drug"), "procedure": ev.get("procedure")}.get(t, t) or "?"


def fields(ev: dict) -> dict:
    """Flatten an event into scoreable fields."""
    t = ev.get("event_type")
    out: dict = {}
    if t == "medication":
        out["drug"] = ev.get("drug")
        out["unit"] = ev.get("unit")
        out["route"] = ev.get("route")
        dose = ev.get("dose")
        if dose is not None and ev.get("drug") in BOOK.p.drugs:
            try:  # compare quantity in canonical unit: 0.1 mg fentanyl == 100 mcg
                dose = to_canonical(dose, ev.get("unit") or BOOK.p.drugs[ev["drug"]].canonical_unit,
                                    BOOK.p.drugs[ev["drug"]].canonical_unit)
            except (ValueError, KeyError):
                pass
        out["dose"] = dose
    elif t == "procedure":
        out["procedure"] = ev.get("procedure")
        if "joules" in (ev.get("params") or {}):
            out["joules"] = ev["params"]["joules"]
    elif t == "vital":
        for k, v in (ev.get("vital") or {}).items():
            out[f"vital.{k}"] = v
    if t != "note":
        out["time"] = ev.get("time_said")
    return {k: v for k, v in out.items() if v is not None}


def same(field: str, g, p, time_tol_s: float) -> bool:
    if field == "time":
        dg, dp = datetime.fromisoformat(g), datetime.fromisoformat(p)
        return abs((dg - dp).total_seconds()) <= time_tol_s
    if isinstance(g, (int, float)) and isinstance(p, (int, float)):
        return math.isclose(float(g), float(p), rel_tol=1e-6, abs_tol=1e-9)
    return g == p


def match_events(gold: list[dict], pred: list[dict]) -> tuple[list[tuple], list[dict], list[dict]]:
    """Pass 1: same type + same key (drug/procedure). Pass 2: same type, in order."""
    g_left, p_left, pairs = list(gold), list(pred), []
    for strict in (True, False):
        for g in list(g_left):
            for p in p_left:
                if p.get("event_type") == g.get("event_type") and (not strict or event_key(p) == event_key(g)):
                    pairs.append((g, p)); g_left.remove(g); p_left.remove(p)
                    break
    return pairs, g_left, p_left


class PRF:
    def __init__(self):
        self.tp = self.fp = self.fn = 0

    def add(self, tp=0, fp=0, fn=0):
        self.tp += tp; self.fp += fp; self.fn += fn

    @property
    def p(self): return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
    @property
    def r(self): return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
    @property
    def f1(self): return 2 * self.p * self.r / (self.p + self.r) if self.p + self.r else 0.0
    @property
    def support(self): return self.tp + self.fn


def score_extraction(gold_rows, pred_rows, time_tol_s=60):
    preds = {r["id"]: r.get("events", []) for r in pred_rows}
    per_field: dict[str, PRF] = defaultdict(PRF)
    by_tag: dict[str, dict[str, PRF]] = defaultdict(lambda: defaultdict(PRF))
    detection = PRF()
    exact = 0
    exact_by_tag: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    errors = []
    missing = [g["id"] for g in gold_rows if g["id"] not in preds]

    for case in gold_rows:
        if case["id"] not in preds:
            continue
        pairs, g_un, p_un = match_events(case["events"], preds[case["id"]])
        case_ok = not g_un and not p_un
        buckets = [per_field] + [by_tag[t] for t in case["tags"]]
        detection.add(tp=len(pairs), fp=len(p_un), fn=len(g_un))
        for b in buckets[1:]:
            b["_events"].add(tp=len(pairs), fp=len(p_un), fn=len(g_un))

        def tally(gf: dict, pf: dict):
            nonlocal case_ok
            for k in set(gf) | set(pf):
                g, p = gf.get(k), pf.get(k)
                ok = g is not None and p is not None and same(k, g, p, time_tol_s)
                tp, fp, fn = int(ok), int(p is not None and not ok), int(g is not None and not ok)
                if not ok:
                    case_ok = False
                    errors.append((case["id"], k, g, p))
                for b in buckets:
                    b[k].add(tp, fp, fn)

        for g, p in pairs:
            tally(fields(g), fields(p))
        for g in g_un:
            tally(fields(g), {})
            errors.append((case["id"], "MISSED_EVENT", event_key(g), None))
        for p in p_un:
            tally({}, fields(p))
            errors.append((case["id"], "EXTRA_EVENT", None, event_key(p)))
        exact += case_ok
        for t in case["tags"]:
            exact_by_tag[t][0] += case_ok
            exact_by_tag[t][1] += 1

    headline = PRF()
    for f in HEADLINE:
        x = per_field.get(f)
        if x:
            headline.add(x.tp, x.fp, x.fn)
    n = len(gold_rows) - len(missing)
    return {"per_field": per_field, "by_tag": by_tag, "detection": detection, "headline": headline,
            "exact_match": exact / n if n else 0.0, "exact_by_tag": exact_by_tag, "n_cases": n, "missing": missing, "errors": errors}


# ------------------------------------------------------------------ ASR
def _norm_words(s: str) -> list[str]:
    s = words_to_numbers(s)
    s = re.sub(r"[^\w\s.:]", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)
    return s.split()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _drug_in(hyp_norm: str, drug: str) -> bool:
    aliases = [a.lower() for a in BOOK.p.drugs[drug].aliases] + [drug.replace("_", " ")]
    return any(re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", hyp_norm) for a in aliases)


def score_asr(gold_rows, asr_rows):
    gold = {g["id"]: g for g in gold_rows}
    stats = defaultdict(lambda: {"edits": 0, "ref_words": 0, "drug_hit": 0, "drug_n": 0,
                                 "dose_hit": 0, "dose_n": 0, "n": 0, "audio_s": 0.0, "proc_s": 0.0})
    worst = []
    for r in asr_rows:
        g = gold.get(r["id"])
        if not g:
            continue
        s = stats[r.get("condition", "default")]
        ref, hyp = _norm_words(g["text"]), _norm_words(r["text"])
        e = edit_distance(ref, hyp)
        s["edits"] += e; s["ref_words"] += len(ref); s["n"] += 1
        s["audio_s"] += r.get("audio_s", 0.0); s["proc_s"] += r.get("latency_s", 0.0)
        hyp_str, hyp_set = " ".join(hyp), set(hyp)
        for ev in g["events"]:
            if ev["event_type"] != "medication":
                continue
            s["drug_n"] += 1; s["drug_hit"] += _drug_in(hyp_str, ev["drug"])
            s["dose_n"] += 1
            d = ev["dose"]
            s["dose_hit"] += (str(int(d)) if d == int(d) else repr(d)) in hyp_set
        worst.append((e / max(len(ref), 1), r.get("condition", "default"), r["id"], g["text"], r["text"]))
    worst.sort(reverse=True)
    return stats, worst[:8]


# ------------------------------------------------------------------ report
def pct(x): return f"{100 * x:5.1f}"


def report_extraction(res, show_tags=True, show_errors=15):
    print("\n=== EXTRACTION ===")
    print(f"cases scored: {res['n_cases']}" + (f"   (missing predictions: {len(res['missing'])})" if res["missing"] else ""))
    print(f"{'field':<16}{'P':>7}{'R':>7}{'F1':>7}{'support':>9}")
    order = HEADLINE + ["procedure", "joules"] + sorted(k for k in res["per_field"] if k.startswith("vital."))
    for k in order:
        x = res["per_field"].get(k)
        if x and (x.support or x.fp):
            print(f"{k:<16}{pct(x.p):>7}{pct(x.r):>7}{pct(x.f1):>7}{x.support:>9}")
    d, h = res["detection"], res["headline"]
    print("-" * 46)
    print(f"{'event detection':<16}{pct(d.p):>7}{pct(d.r):>7}{pct(d.f1):>7}{d.support:>9}")
    print(f"{'HEADLINE F1':<16}{'':>14}{pct(h.f1):>7}   (micro over {', '.join(HEADLINE)})")
    print(f"{'exact-match':<16}{'':>14}{pct(res['exact_match']):>7}   (whole utterance perfect)")

    if show_tags:
        print(f"\n{'by tag':<22}{'headline F1':>12}{'event F1':>10}{'gold ev':>9}{'extra ev':>10}{'exact':>8}")
        for tag, b in sorted(res["by_tag"].items()):
            hp = PRF()
            for f in HEADLINE:
                if f in b:
                    hp.add(b[f].tp, b[f].fp, b[f].fn)
            hl = pct(hp.f1) if (hp.support or hp.fp) else "   - "
            ev = b["_events"]
            ef = pct(ev.f1) if ev.support else "   - "
            ok, n = res["exact_by_tag"][tag]
            print(f"{tag:<22}{hl:>12}{ef:>10}{ev.support:>9}{ev.fp:>10}{pct(ok / n) if n else '  -':>8}")

    if show_errors and res["errors"]:
        print(f"\nfirst {show_errors} errors (case, field, gold, pred):")
        for e in res["errors"][:show_errors]:
            print("  ", e)


def report_asr(stats, worst):
    print("\n=== ASR ===")
    print(f"{'condition':<16}{'WER':>7}{'drug recall':>13}{'dose recall':>13}{'RTF':>7}{'utts':>6}")
    for cond, s in sorted(stats.items()):
        wer = s["edits"] / max(s["ref_words"], 1)
        dr = s["drug_hit"] / max(s["drug_n"], 1)
        do = s["dose_hit"] / max(s["dose_n"], 1)
        rtf = f"{s['proc_s'] / s['audio_s']:.2f}" if s["audio_s"] else "  -"
        print(f"{cond:<16}{pct(wer):>7}{pct(dr):>13}{pct(do):>13}{rtf:>7}{s['n']:>6}")
    print("(drug/dose recall = did the safety-critical words survive ASR; RTF = processing time / audio time)")
    print("\nworst utterances:")
    for w, cond, cid, ref, hyp in worst:
        print(f"  {cid} [{cond}] WER {pct(w)}\n     ref: {ref}\n     hyp: {hyp}")


def append_history(label, res=None, asr_stats=None):
    path = ROOT / "eval" / "results" / "history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "label": label}
    if res:
        ok, n = res["exact_by_tag"].get("hard_heldout", (0, 0))
        row.update({"headline_f1": round(res["headline"].f1, 4), "event_f1": round(res["detection"].f1, 4),
                    "exact_match": round(res["exact_match"], 4),
                    "hard_exact": round(ok / n, 4) if n else ""})
        for f in HEADLINE:
            if f in res["per_field"]:
                row[f"{f}_f1"] = round(res["per_field"][f].f1, 4)
    if asr_stats:
        for cond, s in asr_stats.items():
            row[f"wer_{cond}"] = round(s["edits"] / max(s["ref_words"], 1), 4)
    cols = ["timestamp", "label", "headline_f1", "event_f1", "exact_match", "hard_exact"] + [f"{f}_f1" for f in HEADLINE]
    cols += sorted(k for k in row if k.startswith("wer_"))
    new = not path.exists()
    if not new:  # keep any older columns
        with open(path, encoding="utf-8") as f:
            old_cols = next(csv.reader(f), [])
        cols = old_cols + [c for c in cols if c not in old_cols]
    rows = list(csv.DictReader(open(path, encoding="utf-8"))) if not new else []
    rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nlogged '{label}' -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "eval" / "data" / "gold.jsonl"))
    ap.add_argument("--pred", help="extractor output jsonl: {id, events}")
    ap.add_argument("--asr", help="ASR output jsonl: {id, condition, text}")
    ap.add_argument("--only-recorded", action="store_true", help="score only cases in the recording subset")
    ap.add_argument("--time-tol", type=float, default=60, help="seconds of slack for time_said")
    ap.add_argument("--label", help="append this run to eval/results/history.csv")
    ap.add_argument("--errors", type=int, default=15)
    a = ap.parse_args()
    if not (a.pred or a.asr):
        ap.error("give --pred and/or --asr")

    gold = load_jsonl(a.gold)
    res = stats = None
    if a.pred:
        pred = load_jsonl(a.pred)
        g = gold
        if a.only_recorded:
            g = [x for x in g if x.get("record")]
        if a.asr:  # ASR-based preds only exist for recorded cases
            ids = {p["id"] for p in pred}
            g = [x for x in g if x["id"] in ids]
        res = score_extraction(g, pred, a.time_tol)
        report_extraction(res, show_errors=a.errors)
    if a.asr:
        stats, worst = score_asr(gold, load_jsonl(a.asr))
        report_asr(stats, worst)
    if a.label:
        append_history(a.label, res, stats)


if __name__ == "__main__":
    main()