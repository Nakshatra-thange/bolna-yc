
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
from rule_extractor import ROUTE_PATTERNS, UNIT_PATTERNS  # noqa: E402

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


def check_preds(gold_rows, pred_rows) -> None:
    """Raise on prediction files that would score as something they're not."""
    if not pred_rows:
        raise SystemExit("prediction file is empty - nothing to score")
    gold_ids = {g["id"] for g in gold_rows}
    ids = [r.get("id") for r in pred_rows]
    if None in ids or any("events" not in r for r in pred_rows):
        raise SystemExit("prediction rows must have 'id' and 'events'")
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise SystemExit(f"duplicate prediction ids {dup} (mixed conditions? extract one --condition at a time)")
    unknown = sorted(set(ids) - gold_ids)
    if unknown:
        raise SystemExit(f"prediction ids not in gold: {unknown}")
    conds = {r.get("condition") for r in pred_rows}
    if len(conds) > 1:
        raise SystemExit(f"prediction file mixes conditions {sorted(map(str, conds))}")


def score_extraction(gold_rows, pred_rows, time_tol_s=60):
    preds = {r["id"]: r.get("events", []) for r in pred_rows}
    if not any(g["id"] in preds for g in gold_rows):
        raise ValueError("0 gold cases have a prediction - refusing to report an empty evaluation")
    per_field: dict[str, PRF] = defaultdict(PRF)
    by_tag: dict[str, dict[str, PRF]] = defaultdict(lambda: defaultdict(PRF))
    detection = PRF()
    exact = 0
    exact_by_tag: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    safety = {"meds": 0, "flagged": 0, "wrong": 0, "wrong_flagged": 0}
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

        def safety_count(p: dict, wrong: bool):
            if p.get("event_type") != "medication":
                return
            flagged = bool(p.get("needs_confirmation"))
            safety["meds"] += 1
            safety["flagged"] += flagged
            safety["wrong"] += wrong
            safety["wrong_flagged"] += wrong and flagged

        for g, p in pairs:
            gf, pf = fields(g), fields(p)
            safety_count(p, any(gf.get(k) is None or pf.get(k) is None or not same(k, gf[k], pf[k], time_tol_s)
                                for k in set(gf) | set(pf)))
            tally(gf, pf)
        for g in g_un:
            tally(fields(g), {})
            errors.append((case["id"], "MISSED_EVENT", event_key(g), None))
        for p in p_un:
            safety_count(p, True)
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
            "exact_match": exact / n if n else 0.0, "exact_by_tag": exact_by_tag, "safety": safety, "n_cases": n, "missing": missing, "errors": errors}


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


def check_asr(gold_rows, asr_rows, field="text") -> None:
    """Raise on ASR files that can't be scored honestly (no silent fallbacks or skips)."""
    if not asr_rows:
        raise SystemExit("ASR file has no rows (after condition filter)")
    gold_ids = {g["id"] for g in gold_rows}
    bad = [f"{r.get('id')} [{r.get('condition')}]" for r in asr_rows if field not in r or "condition" not in r]
    if bad:
        raise SystemExit(f"ASR rows lack {field!r} or 'condition': {bad[:5]}{' ...' if len(bad) > 5 else ''}")
    keys = [(r["condition"], r["id"]) for r in asr_rows]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise SystemExit(f"duplicate (condition, id) rows: {dup[:5]}")
    unknown = sorted({r["id"] for r in asr_rows} - gold_ids)
    if unknown:
        raise SystemExit(f"ASR ids not in gold: {unknown}")


def _safety_checks(ev: dict, hyp_str: str, hyp_set: set) -> dict[str, bool]:
    """Did each safety-critical part of a gold medication survive ASR? (word-level, no extraction)"""
    d = ev["dose"]
    out = {"drug": _drug_in(hyp_str, ev["drug"]),
           "dose": (str(int(d)) if d == int(d) else repr(d)) in hyp_set}
    if ev.get("unit"):
        out["unit"] = bool(re.search(r"(?<![a-z])(?:" + dict(UNIT_PATTERNS)[ev["unit"]] + r")(?![a-z])", hyp_str))
    if ev.get("route"):
        out["route"] = bool(re.search(dict(ROUTE_PATTERNS)[ev["route"]], hyp_str))
    return out


SAFETY_FIELDS = ["drug", "dose", "unit", "route"]


def score_asr(gold_rows, asr_rows, field="text"):
    """Returns (stats per condition, per-utterance detail rows)."""
    gold = {g["id"]: g for g in gold_rows}
    stats = defaultdict(lambda: {"edits": 0, "ref_words": 0, "n": 0, "audio_s": 0.0, "proc_s": 0.0,
                                 **{f"{k}_{x}": 0 for k in SAFETY_FIELDS for x in ("hit", "n")}})
    detail = []
    for r in asr_rows:
        g = gold[r["id"]]
        s = stats[r["condition"]]
        hyp_text = r[field]
        ref, hyp = _norm_words(g["text"]), _norm_words(hyp_text)
        e = edit_distance(ref, hyp)
        s["edits"] += e; s["ref_words"] += len(ref); s["n"] += 1
        s["audio_s"] += r.get("audio_s", 0.0); s["proc_s"] += r.get("latency_s", 0.0)
        hyp_str, hyp_set = " ".join(hyp), set(hyp)
        missed = []
        for ev in g["events"]:
            if ev["event_type"] != "medication":
                continue
            for k, ok in _safety_checks(ev, hyp_str, hyp_set).items():
                s[f"{k}_n"] += 1; s[f"{k}_hit"] += ok
                if not ok:
                    missed.append(f"{k}:{ev[k]}")
        detail.append({"wer": e / max(len(ref), 1), "condition": r["condition"], "id": r["id"],
                       "ref": g["text"], "hyp": hyp_text, "raw": r.get("raw_text"), "missed": missed,
                       "corrections": r.get("corrections", [])})
    return stats, detail


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

    sf = res["safety"]
    if sf["meds"]:
        unflagged = sf["wrong"] - sf["wrong_flagged"]
        print(f"\nSAFETY (medication events): {sf['meds']} recorded, {sf['flagged']} sent for confirmation "
              f"({pct(sf['flagged'] / sf['meds']).strip()}%)")
        print(f"  wrong: {sf['wrong']}   caught by confirmation: {sf['wrong_flagged']}   "
              f"WRONG AND UNFLAGGED: {unflagged} ({pct(unflagged / sf['meds']).strip()}% of recorded meds)")

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


SYNTHETIC = {"tts"}  # reported apart so it can't hide (or flatter) the recorded-audio numbers


def _asr_row(name, s):
    wer = s["edits"] / max(s["ref_words"], 1)
    rec = "".join(f"{pct(s[f'{k}_hit'] / s[f'{k}_n']) if s[f'{k}_n'] else '   - ':>8}" for k in SAFETY_FIELDS)
    rtf = f"{s['proc_s'] / s['audio_s']:.2f}" if s["audio_s"] else "  -"
    print(f"{name:<16}{pct(wer):>7}{rec}{rtf:>7}{s['n']:>6}")


def _print_detail(d):
    print(f"  {d['id']} [{d['condition']}] WER {pct(d['wer'])}   missed: {', '.join(d['missed']) or 'none'}")
    print(f"     ref: {d['ref']}\n     hyp: {d['hyp']}")
    if d["raw"] is not None and d["raw"] != d["hyp"]:
        print(f"     raw: {d['raw']}")
    for c in d["corrections"]:
        print(f"     lexicon: {c['heard']!r} -> {c['fixed']!r} ({c['kind']}, {c['score']})")


def report_asr(stats, detail, field="text", show_ids=(), n_worst=8):
    print(f"\n=== ASR (field: {field}) ===")
    print(f"{'condition':<16}{'WER':>7}" + "".join(f"{k + ' R':>8}" for k in SAFETY_FIELDS) + f"{'RTF':>7}{'utts':>6}")
    groups = [("recorded", sorted(c for c in stats if c not in SYNTHETIC)),
              ("synthetic", sorted(c for c in stats if c in SYNTHETIC))]
    for gname, conds in groups:
        if not conds:
            continue
        for cond in conds:
            _asr_row(cond, stats[cond])
        if len(conds) > 1:
            pool = defaultdict(float)
            for c in conds:
                for k, v in stats[c].items():
                    pool[k] += v
            _asr_row(f"[{gname} pooled]", pool)
        print("-" * 70)
    print("(R = recall of the gold medication's drug / dose / unit / route words in the transcript;\n"
          " RTF = processing time / audio time)")
    for gname, conds in groups:
        rows = sorted((d for d in detail if d["condition"] in conds), key=lambda d: -d["wer"])
        if rows:
            print(f"\nworst {gname} utterances:")
            for d in rows[:n_worst]:
                _print_detail(d)
    for cid in show_ids:
        rows = [d for d in detail if d["id"] == cid]
        print(f"\n--- {cid}: {len(rows)} condition(s) ---")
        for d in rows:
            _print_detail(d)


def append_history(label, res=None, asr_stats=None, sources=None):
    path = ROOT / "eval" / "results" / "history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "label": label, **(sources or {})}
    if res:
        ok, n = res["exact_by_tag"].get("hard_heldout", (0, 0))
        row.update({"n_cases": res["n_cases"],"headline_f1": round(res["headline"].f1, 4), "event_f1": round(res["detection"].f1, 4),
                    "exact_match": round(res["exact_match"], 4),
                    "hard_exact": round(ok / n, 4) if n else "",
                    "unflagged_wrong_meds": res["safety"]["wrong"] - res["safety"]["wrong_flagged"],
                    "flag_rate": round(res["safety"]["flagged"] / res["safety"]["meds"], 4) if res["safety"]["meds"] else ""})
        for f in HEADLINE:
            if f in res["per_field"]:
                row[f"{f}_f1"] = round(res["per_field"][f].f1, 4)
    if asr_stats:
        for cond, s in asr_stats.items():
            row[f"wer_{cond}"] = round(s["edits"] / max(s["ref_words"], 1), 4)
            row[f"n_{cond}"] = s["n"]
    cols = ["timestamp", "label", "pred_file", "asr_file", "asr_field", "condition", "n_cases", "headline_f1", "event_f1", "exact_match", "hard_exact",
            "unflagged_wrong_meds", "flag_rate"] + [f"{f}_f1" for f in HEADLINE]
    cols += sorted(k for k in row if k.startswith(("wer_", "n_")) and k != "n_cases")
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


def load_asr(path, field, conditions=None) -> list[dict]:
    rows = load_jsonl(path)
    if conditions:
        have = sorted({r.get("condition") for r in rows})
        absent = [c for c in conditions if c not in have]
        if absent:
            raise SystemExit(f"{path}: no rows for condition(s) {absent}; file has {have}")
        rows = [r for r in rows if r.get("condition") in conditions]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "eval" / "data" / "gold.jsonl"))
    ap.add_argument("--pred", help="extractor output jsonl: {id, events[, condition]}")
    ap.add_argument("--asr", help="ASR output jsonl: {id, condition, text, raw_text}")
    ap.add_argument("--asr-field", default="text", help="'text' (after lexicon fix) or 'raw_text' (straight ASR)")
    ap.add_argument("--conditions", nargs="*", help="only score these ASR conditions (e.g. --conditions tts)")
    ap.add_argument("--same-utts-as", help="another ASR jsonl; fail unless both cover identical (condition, id) sets")
    ap.add_argument("--show-ids", nargs="*", default=[], help="print every condition's transcript for these ids")
    ap.add_argument("--only-recorded", action="store_true", help="score only cases in the recording subset")
    ap.add_argument("--time-tol", type=float, default=60, help="seconds of slack for time_said")
    ap.add_argument("--label", help="append this run to eval/results/history.csv")
    ap.add_argument("--errors", type=int, default=15)
    a = ap.parse_args()
    if not (a.pred or a.asr):
        ap.error("give --pred and/or --asr")

    gold = load_jsonl(a.gold)
    res = stats = None
    sources = {}
    pred_cond = None
    if a.pred:
        pred = load_jsonl(a.pred)
        check_preds(gold, pred)
        pred_cond = pred[0].get("condition")
        g = gold
        if a.only_recorded:
            g = [x for x in g if x.get("record")]
        if a.asr or pred_cond:  # ASR-based preds only exist for recorded cases
            ids = {p["id"] for p in pred}
            g = [x for x in g if x["id"] in ids]
        print(f"extraction: {len(pred)} predictions from {a.pred}"
              + (f" (condition {pred_cond})" if pred_cond else "") + f", {len(g)} gold cases matched")
        res = score_extraction(g, pred, a.time_tol)
        report_extraction(res, show_errors=a.errors)
        sources.update(pred_file=Path(a.pred).name, condition=pred_cond or "")
    if a.asr:
        conds = a.conditions
        if pred_cond:  # the ASR numbers logged next to an extraction score must be for the same audio
            if conds and conds != [pred_cond]:
                raise SystemExit(f"--conditions {conds} disagrees with prediction condition {pred_cond!r}")
            conds = [pred_cond]
        asr = load_asr(a.asr, a.asr_field, conds)
        check_asr(gold, asr, a.asr_field)
        if pred_cond and {r["id"] for r in asr} != {p["id"] for p in pred}:
            raise SystemExit(f"ASR ids for {pred_cond} differ from prediction ids")
        if a.same_utts_as:
            other = load_asr(a.same_utts_as, a.asr_field, conds)
            mine, theirs = {(r["condition"], r["id"]) for r in asr}, {(r["condition"], r.get("id")) for r in other}
            if mine != theirs:
                raise SystemExit(f"utterance sets differ from {a.same_utts_as}: only here {sorted(mine - theirs)[:5]}, "
                                 f"only there {sorted(theirs - mine)[:5]}")
            print(f"same utterances as {a.same_utts_as}: {len(mine)} (condition, id) pairs")
        stats, detail = score_asr(gold, asr, a.asr_field)
        report_asr(stats, detail, a.asr_field, a.show_ids)
        sources.update(asr_file=Path(a.asr).name, asr_field=a.asr_field)
    if a.label:
        append_history(a.label, res, stats, sources)


if __name__ == "__main__":
    main()
