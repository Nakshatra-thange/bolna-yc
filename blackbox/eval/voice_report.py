"""
Black Box - voice-agent evaluation report (organised like the five dimensions in Hamming's metrics reference:
ASR accuracy, latency, task/intent success, safety, and what the metrics don't capture).

  python eval/run_extractor.py --extractor rules                 # once (makes eval/preds/ext_rules.jsonl)
  python eval/voice_report.py                                    # extraction + intent + safety + latency (no audio needed)
  python eval/voice_report.py --asr eval/preds/asr_fw-base_prompt.jsonl     # + WER / CER / entity accuracy / RTF per condition
  python eval/voice_report.py --pred eval/preds/ext_hybrid_<model>.jsonl --label hybrid

Everything is measured from files in this repo; nothing is hard-coded. The small hand-written command set is a
sanity check (n is printed), not a benchmark.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from score import _drug_in, _norm_words, _raw_words, edit_distance, score_asr, score_extraction  # noqa: E402

from extract.confirm import parse_command  # noqa: E402


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def pctile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else float("nan")


def head(t):
    print(f"\n{'=' * 4} {t} {'=' * max(4, 66 - len(t))}")


# --------------------------------------------------------------------------- command intents
# Hand-written, labelled by what a human medic MEANS (not by what the parser returns). None = not a command.
COMMANDS = [
    ("confirm", "confirm"), ("confirm", "yes"), ("confirm", "correct"), ("confirm", "that's right"),
    ("confirm", "haan"), ("confirm", "theek hai"), ("confirm", "sahi hai"), ("confirm", "yes confirm it"),
    ("confirm", "ok confirm"), ("confirm", "yeah that is correct"), ("confirm", "confirmed"), ("confirm", "right"),
    ("cancel", "cancel"), ("cancel", "no"), ("cancel", "wrong"), ("cancel", "scratch that"), ("cancel", "galat"),
    ("cancel", "nahi"), ("cancel", "cancel karo"), ("cancel", "no that is wrong"), ("cancel", "negative"),
    ("cancel", "never mind"), ("cancel", "stop that one"), ("cancel", "delete"),
    (None, "gave adrenaline one milligram"), (None, "patient weight seventy kilos"), (None, "blood pressure ninety over sixty"),
    (None, "I'm not sure"), (None, "give the confirmation later"), (None, "the patient says no allergies"),
    (None, "wrong route on the last one but keep the dose"), (None, "okay moving on to the next patient"),
    (None, "yes the patient is awake"), (None, "cancel the ambulance request"), (None, "defibrillate at two hundred joules"),
    (None, "adrenaline ek milligram IV diya"), (None, "haan patient ko allergy hai"), (None, "no pulse"),
]


def command_report():
    head("INTENT: spoken commands (confirm / cancel / not-a-command)")
    cm = Counter()
    wrong = []
    for gold, text in COMMANDS:
        pred = parse_command(text)
        cm[(gold, pred)] += 1
        if gold != pred:
            wrong.append((gold, pred, text))
    labels = ["confirm", "cancel", None]
    n = len(COMMANDS)
    correct = sum(v for (g, p), v in cm.items() if g == p)
    print(f"n={n} hand-written utterances (English + Hinglish)   accuracy {pct(correct / n)}")
    print(f"{'':>14}" + "".join(f"{('pred ' + str(l)):>14}" for l in labels))
    for g in labels:
        print(f"{'gold ' + str(g):>14}" + "".join(f"{cm[(g, p)]:>14}" for p in labels))
    false_confirm = sum(v for (g, p), v in cm.items() if p == "confirm" and g != "confirm")
    false_cancel = sum(v for (g, p), v in cm.items() if p == "cancel" and g != "cancel")
    print(f"WRONG 'confirm' (would record an unwanted event as confirmed): {false_confirm}")
    print(f"WRONG 'cancel'  (would drop a good event):                      {false_cancel}")
    print("Design choice: the parser is deliberately strict (short utterances only) so it misses some natural phrasings "
          "rather than confirming by accident.")
    for g, p, t in wrong:
        print(f"   missed/wrong: {t!r}: gold {g}, parsed {p}")
    return {"n": n, "accuracy": correct / n, "false_confirm": false_confirm, "false_cancel": false_cancel}


# --------------------------------------------------------------------------- extraction + intent on gold text
def intent_of(events):
    return "+".join(sorted({e["event_type"] for e in events})) or "none"


def extraction_report(gold, preds, label):
    pmap = {p["id"]: p for p in preds}
    gold = [g for g in gold if g["id"] in pmap]
    res = score_extraction(gold, preds)
    out = {}

    head(f"TASK SUCCESS: structured extraction ({label}, text input = gold transcripts)")
    print(f"{'slice':<16}{'n':>5}{'headline F1':>13}{'event F1':>10}{'exact-match':>13}")
    rows = [("ALL", None)] + [(t, t) for t in ("english", "hinglish", "hard_heldout", "brand_name", "numbers_colloquial", "time_relative")]
    for name, tag in rows:
        sub = [g for g in gold if tag is None or tag in g["tags"]]
        if not sub:
            continue
        r = score_extraction(sub, [pmap[g["id"]] for g in sub])
        print(f"{name:<16}{len(sub):>5}{pct(r['headline'].f1):>13}{pct(r['detection'].f1):>10}{pct(r['exact_match']):>13}")
        out[name] = {"n": len(sub), "headline_f1": r["headline"].f1, "event_f1": r["detection"].f1, "exact": r["exact_match"]}
    print("(Hinglish and hard_heldout are the honest numbers; ALL is flattered by the easy English cases.)")

    head("INTENT: what is the speaker doing? (utterance-level event type)")
    cm = Counter()
    for g in gold:
        cm[(intent_of(g["events"]), intent_of(pmap[g["id"]]["events"]))] += 1
    n = len(gold)
    acc = sum(v for (a, b), v in cm.items() if a == b) / n
    print(f"n={n}   intent accuracy {pct(acc)}   (exact set of event types per utterance)")
    print(f"{'type':<14}{'precision':>11}{'recall':>9}{'support':>9}")
    for t in ("medication", "vital", "procedure"):
        tp = sum(1 for g in gold if t in intent_of(g["events"]).split("+") and t in intent_of(pmap[g["id"]]["events"]).split("+"))
        fp = sum(1 for g in gold if t not in intent_of(g["events"]).split("+") and t in intent_of(pmap[g["id"]]["events"]).split("+"))
        fn = sum(1 for g in gold if t in intent_of(g["events"]).split("+") and t not in intent_of(pmap[g["id"]]["events"]).split("+"))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        print(f"{t:<14}{pct(p):>11}{pct(r):>9}{tp + fn:>9}")
    bad = Counter((a, b) for (a, b), v in cm.items() for _ in range(v) if a != b)
    if bad:
        print("confusions (gold -> predicted):", dict(bad))
    out["intent_accuracy"] = acc

    head("SAFETY: errors that matter clinically")
    sf = res["safety"]
    unflagged = sf["wrong"] - sf["wrong_flagged"]
    print(f"medication events recorded:            {sf['meds']}")
    print(f"  wrong (any field):                   {sf['wrong']}")
    print(f"  wrong AND caught by confirmation:    {sf['wrong_flagged']}")
    print(f"  WRONG AND UNFLAGGED:                 {unflagged}  ({pct(unflagged / max(sf['meds'], 1))} of recorded meds)   <- the number to drive down")
    print(f"  confirmation load (events sent to the medic): {pct(sf['flagged'] / max(sf['meds'], 1))}   <- the cost of the safety net")
    # unsupported-value rate: a predicted drug/dose that is not in the transcript at all (hallucination proxy)
    bad_n = tot = 0
    for g in gold:
        toks, txt = set(_norm_words(g["text"])), " ".join(_norm_words(g["text"]))
        for e in pmap[g["id"]]["events"]:
            if e.get("event_type") != "medication":
                continue
            tot += 1
            d = e.get("dose")
            dose_ok = d is None or (str(int(d)) if d == int(d) else repr(d)) in toks
            drug_ok = bool(e.get("drug")) and _drug_in(txt, e["drug"])
            bad_n += not (dose_ok and drug_ok)
    print(f"unsupported drug/dose (not present in transcript): {bad_n}/{tot} = {pct(bad_n / max(tot, 1), 2)}   "
          f"(hallucination proxy; the verify-in-code layer is meant to keep this at 0)")
    out["safety"] = {**sf, "unflagged": unflagged, "unsupported": bad_n, "meds_checked": tot}
    out["errors_top"] = [str(e) for e in res["errors"][:8]]
    return out


def latency_report(preds):
    head("LATENCY: text stage (extraction)")
    lat = [1000 * p["meta"]["latency_s"] for p in preds if p.get("meta", {}).get("latency_s") is not None]
    if lat:
        print(f"n={len(lat)}   p50 {pctile(lat, .5):.1f} ms   p95 {pctile(lat, .95):.1f} ms   max {max(lat):.1f} ms")
        print("(Rules are near-instant; an LLM adds hundreds of ms to seconds. Speech recognition dominates end-to-end time, see ASR section.)")
    return {"p50_ms": pctile(lat, .5), "p95_ms": pctile(lat, .95)} if lat else {}


# --------------------------------------------------------------------------- ASR (needs audio run)
def asr_report(gold, asr_rows, label):
    head(f"ASR ACCURACY + SPEED ({label})")
    stats, worst = score_asr(gold, asr_rows)
    gmap = {g["id"]: g for g in gold}
    cer = defaultdict(lambda: [0, 0])
    runaway = defaultdict(lambda: [0, 0])
    lat = defaultdict(list)
    for r in asr_rows:
        g = gmap.get(r["id"])
        if not g:
            continue
        c = r.get("condition", "default")
        ref, hyp = " ".join(_norm_words(g["text"])), " ".join(_norm_words(r["text"]))
        cer[c][0] += edit_distance(list(ref), list(hyp))
        cer[c][1] += len(ref)
        lat[c].append(r.get("latency_s", 0.0))
        runaway[c][0] += len(hyp.split()) > 2 * len(ref.split()) + 5      # output far longer than what was said
        runaway[c][1] += 1
    print(f"{'condition':<18}{'n':>4}{'raw WER':>9}{'norm WER':>10}{'norm CER':>10}{'drug ok':>9}{'dose ok':>9}{'runaway':>9}{'RTF':>7}{'p50 s':>8}{'p95 s':>8}")
    out = {}
    for c, s in sorted(stats.items()):
        wer = s["edits"] / max(s["ref_words"], 1)
        raw_wer = s["raw_edits"] / max(s["raw_ref_words"], 1)
        cr = cer[c][0] / max(cer[c][1], 1)
        rtf = s["proc_s"] / s["audio_s"] if s["audio_s"] else float("nan")
        print(f"{c:<18}{s['n']:>4}{pct(raw_wer):>9}{pct(wer):>10}{pct(cr):>10}{pct(s['drug_hit'] / max(s['drug_n'], 1)):>9}"
              f"{pct(s['dose_hit'] / max(s['dose_n'], 1)):>9}{pct(runaway[c][0] / max(runaway[c][1], 1), 0):>9}{rtf:>7.2f}{pctile(lat[c], .5):>8.2f}{pctile(lat[c], .95):>8.2f}")
        out[c] = {"n": s["n"], "raw_wer": raw_wer, "wer": wer, "cer": cr, "drug_acc": s["drug_hit"] / max(s["drug_n"], 1),
                  "dose_acc": s["dose_hit"] / max(s["dose_n"], 1), "rtf": rtf,
                  "runaway_rate": runaway[c][0] / max(runaway[c][1], 1)}
    # The gold tags overlap (hard_heldout clips may also be Hinglish); show each label as its own slice.
    print("\nTag slices (overlapping labels; percentages use each slice's pooled reference tokens):")
    for tag in ("english", "hinglish", "hard_heldout"):
        tagged = [g for g in gold if tag in g.get("tags", [])]
        ids = {g["id"] for g in tagged}
        # Tag comparison is on the primary clean synthetic set so repeated noisy
        # versions of the same utterance cannot inflate n or dominate the slice.
        rows = [r for r in asr_rows if r["id"] in ids and r.get("condition") == "tts"]
        if not rows:
            continue
        sub_stats, _ = score_asr(tagged, rows)
        s = defaultdict(float)
        for part in sub_stats.values():
            for key, value in part.items():
                s[key] += value
        print(f"  {tag:<14} n={int(s['n']):>3} raw WER={pct(s['raw_edits']/max(s['raw_ref_words'],1))} "
              f"normalized WER={pct(s['edits']/max(s['ref_words'],1))} "
              f"drug ok={pct(s['drug_hit']/max(s['drug_n'],1))} dose ok={pct(s['dose_hit']/max(s['dose_n'],1))}")
    print("Raw WER lowercases and removes punctuation only. Normalized WER additionally folds number words, unit/route synonyms, and common spoken variants. "
          "RTF < 1.0 means faster than real time. 'runaway' = clips whose output is far longer than what was said "
          "(repetition-loop hallucination); it inflates WER above 100% and must be ~0.")
    print("Worst utterances (WER):")
    # score_asr returns detail rows as dicts; be robust to tuple/list too.
    if worst and isinstance(worst[0], dict):
        worst_rows = sorted(worst, key=lambda d: -d.get("wer", 0))
    else:
        worst_rows = list(worst)
    for item in worst_rows[:5]:
        if isinstance(item, dict):
            w = item.get("wer", 0)
            c, i = item.get("condition", "?"), item.get("id", "?")
            ref, hyp = item.get("ref", ""), item.get("hyp", "")
        else:
            w, c, i, *rest = item
            ref, hyp = (rest[-2], rest[-1]) if len(rest) >= 2 else ("", "")
        wv = float(w) if not isinstance(w, (int, float)) else w
        print(f"   {c}/{i}  WER {wv:.2f}\n      ref: {ref}\n      hyp: {str(hyp)[:140]}")
    return out


def reference_table():
    head("HOW THIS MAPS TO THE HAMMING REFERENCE (benchmarks are a vendor's call-centre guidance, not a standard)")
    print("""  Dimension           Their metric                Ours                                   Their healthcare target
  ASR accuracy        WER, CER, entity accuracy   WER + CER + drug/dose accuracy         WER < 5%  (we also care about entities)
  Latency             p50/p95, RTF                RTF + p50/p95 per stage                P95 < 1.2 s (cloud agents; ours is a recorder, not a chat agent)
  Intent accuracy     correct / total             event-type intent + command intent     > 95%
  Task success        TSR                         exact-match + per-field F1             > 85%
  Safety              hallucination rate          unsupported-value rate + WRONG AND UNFLAGGED   < 0.5% / zero propagation
  Not covered         TTS MOS, containment, FCR   n/a: no spoken answers to callers, no call-centre handoff""")
    print("\n  What no metric here captures (their own caveat applies to us too): a low WER can hide a wrong dose; "
          "small hand-built sets; synthetic voices; no real ambulance audio yet.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "eval" / "data" / "gold.jsonl"))
    ap.add_argument("--pred", default=str(ROOT / "eval" / "preds" / "ext_rules.jsonl"))
    ap.add_argument("--label", default="rules only")
    ap.add_argument("--asr", nargs="*", help="eval/preds/asr_*.jsonl written by run_asr.py")
    a = ap.parse_args()

    pred_path = Path(a.pred)
    if not pred_path.exists():
        print(f"{pred_path.name} missing: running the rules extractor first ...")
        subprocess.run([sys.executable, str(ROOT / "eval" / "run_extractor.py"), "--extractor", "rules"], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
    gold, preds = load(a.gold), load(pred_path)

    report = {"label": a.label}
    report["extraction"] = extraction_report(gold, preds, a.label)
    report["commands"] = command_report()
    report["latency_text_stage"] = latency_report(preds)
    for p in a.asr or []:
        report.setdefault("asr", {})[Path(p).stem] = asr_report(gold, load(p), Path(p).stem)
    if not a.asr:
        head("ASR ACCURACY + SPEED")
        print("not run yet. Needs audio:  python eval/run_asr.py --model base   then   python eval/voice_report.py --asr eval/preds/asr_fw-base_prompt.jsonl")
    reference_table()

    out = ROOT / "eval" / "results" / "voice_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n(saved {out.relative_to(ROOT)})")


if __name__ == "__main__":
    main()
