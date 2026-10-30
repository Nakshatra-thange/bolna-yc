"""
Black Box Guardian - run every scripted scene through the SAME code path as the live pipeline
(extraction -> context -> Guardian.assess) and score alerts against expectations.

  python eval/score_guardian.py            # per-scene table + per-rule precision/recall
  python eval/score_guardian.py -v         # also print every alert's spoken text
  python eval/score_guardian.py --label guardian-v1     # append to eval/results/guardian_history.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from extract.hybrid import HybridExtractor  # noqa: E402
from guardian.guardian import Guardian  # noqa: E402
from guardian_scenes import SCENES  # noqa: E402

T0 = datetime(2026, 10, 30, 9, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))


def run_scene(scene: dict, extractor=None) -> list[tuple[int, object]]:
    extractor = extractor or HybridExtractor(llm=None)
    g = Guardian()
    fired = []
    for i, (minute, text) in enumerate(scene["steps"]):
        t = T0 + timedelta(minutes=minute)
        events, _ = extractor.extract(text, t)
        for k, e in enumerate(events):
            e["event_id"] = f"{scene['name']}-{i}-{k}"
        upd = g.observe_context(text)
        _, alerts = g.assess(events, text, upd)
        fired += [(i, a) for a in alerts]
    return fired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--label")
    a = ap.parse_args()

    per_rule = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
    scenes_ok = 0
    false_alarms = 0
    for sc in SCENES:
        fired = run_scene(sc)
        got = {(i, al.rule_id) for i, al in fired}
        exp = sc["expected"]
        for x in got & exp:
            per_rule[x[1]][0] += 1
        for x in got - exp:
            per_rule[x[1]][1] += 1
            false_alarms += 1
        for x in exp - got:
            per_rule[x[1]][2] += 1
        ok = got == exp
        scenes_ok += ok
        print(f"{'PASS' if ok else 'FAIL'}  {sc['name']:<30} expected {sorted(exp) or '-'}"
              + ("" if ok else f"   got {sorted(got) or '-'}"))
        if a.verbose:
            for i, al in fired:
                print(f"        step {i} [{al.severity}] {al.spoken}")

    print(f"\n{'rule':<20}{'precision':>10}{'recall':>8}{'tp':>5}{'fp':>5}{'fn':>5}")
    tp_all = fp_all = fn_all = 0
    for rule, (tp, fp, fn) in sorted(per_rule.items()):
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
        print(f"{rule:<20}{p:>10.2f}{r:>8.2f}{tp:>5}{fp:>5}{fn:>5}")
    P = tp_all / (tp_all + fp_all) if tp_all + fp_all else 0
    R = tp_all / (tp_all + fn_all) if tp_all + fn_all else 0
    print(f"\nscenes passed {scenes_ok}/{len(SCENES)}   alert precision {P:.2f}   recall {R:.2f}   "
          f"false alarms {false_alarms}")

    if a.label:
        path = ROOT / "eval" / "results" / "guardian_history.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "label", "scenes_passed", "scenes", "precision", "recall", "false_alarms"])
            w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), a.label, scenes_ok, len(SCENES),
                        round(P, 4), round(R, 4), false_alarms])
        print(f"logged -> {path}")
    sys.exit(0 if scenes_ok == len(SCENES) else 1)


if __name__ == "__main__":
    main()