
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from protocol import ProtocolBook, to_canonical
from rule_extractor import extract as rule_extract

BOOK = ProtocolBook.load()
ASR_CONF_MIN = 0.60


def _canon(ev: dict) -> Optional[float]:
    d, drug = ev.get("dose"), ev.get("drug")
    if d is None or drug not in BOOK.p.drugs:
        return d
    try:
        return to_canonical(d, ev.get("unit") or BOOK.p.drugs[drug].canonical_unit, BOOK.p.drugs[drug].canonical_unit)
    except ValueError:
        return d


def _flag(ev: dict, reason: str):
    ev.setdefault("confirm_reasons", [])
    if reason not in ev["confirm_reasons"]:
        ev["confirm_reasons"].append(reason)
    ev["needs_confirmation"] = True


class HybridExtractor:
    def __init__(self, llm=None, confirm_controlled: bool = True):
        self.llm = llm
        self.confirm_controlled = confirm_controlled

    def extract(self, text: str, time_logged: datetime, asr: Optional[dict] = None) -> tuple[list[dict], dict]:
        t0 = time.perf_counter()
        rules = rule_extract(text, time_logged)
        meta: dict = {"source": "rules"}
        held: list[dict] = []

        if self.llm is None:
            events = [dict(e, needs_confirmation=False, confirm_reasons=[]) for e in rules]
        else:
            try:
                events, held, m = self.llm.extract(text, time_logged)
                meta.update(m, source="hybrid")
            except Exception as exc:  # noqa: BLE001 - any LLM failure falls back to rules
                meta.update(source="rules-fallback", llm_error=f"{type(exc).__name__}: {exc}")
                events = [dict(e, needs_confirmation=False, confirm_reasons=[]) for e in rules]
                for e in events:
                    if e["event_type"] == "medication":
                        _flag(e, "LLM unavailable - rules only")
            else:
                self._cross_check(events, held, rules)

        for e in events:
            if e["event_type"] != "medication":
                continue
            drug = e.get("drug")
            if e.get("dose") is None:
                _flag(e, "no dose")
            if not e.get("route"):
                _flag(e, "no route")
            if self.confirm_controlled and drug in BOOK.p.drugs and BOOK.p.drugs[drug].controlled:
                _flag(e, "controlled drug")
            if asr:
                if asr.get("asr_confidence") is not None and asr.get("engine", "").startswith("faster-whisper") \
                        and asr["asr_confidence"] < ASR_CONF_MIN:
                    _flag(e, f"low ASR confidence {asr['asr_confidence']:.2f}")
                for c in asr.get("corrections", []):
                    if c.get("kind") == "drug" and BOOK.resolve_drug(c.get("fixed", "")) == drug:
                        _flag(e, f"drug name auto-corrected from '{c['heard']}'")
                    if c.get("kind") == "known" and c.get("fixed", "")[:1].isdigit():
                        _flag(e, f"dose word auto-corrected from '{c['heard']}'")
        meta["held"] = held
        meta["latency_s"] = round(time.perf_counter() - t0, 3)
        return events, meta

    # -------------------------------------------------------------
    @staticmethod
    def _cross_check(events: list[dict], held: list[dict], rules: list[dict]):
        rule_meds = [r for r in rules if r["event_type"] == "medication"]
        llm_meds = [e for e in events if e["event_type"] == "medication"]
        used = set()
        for e in llm_meds:
            match = next((i for i, r in enumerate(rule_meds) if i not in used and r.get("drug") == e.get("drug")), None)
            if match is None:
                continue
            used.add(match)
            r = rule_meds[match]
            rd, ld = _canon(r), _canon(e)
            if rd is not None and ld is not None and abs(rd - ld) > 1e-9:
                _flag(e, f"rules heard dose {r['dose']:g} {r.get('unit', '')}".strip())
            if r.get("unit") and e.get("unit") and r["unit"].lower() != e["unit"].lower():
                _flag(e, f"rules heard unit {r['unit']}")
            if r.get("route") and e.get("route") and r["route"] != e["route"]:
                _flag(e, f"rules heard route {r['route']}")
        held_drugs = {h.get("drug") for h in held}
        for i, r in enumerate(rule_meds):
            if i in used or r.get("drug") in held_drugs:
                continue
            if any(e.get("drug") == r.get("drug") for e in llm_meds):
                continue
            extra = dict(r, needs_confirmation=False, confirm_reasons=[])
            _flag(extra, "only the rule extractor saw this")
            events.append(extra)
        # procedures / vitals the LLM missed: take them from the rules (not doses, so no confirmation needed)
        llm_procs = {e.get("procedure") for e in events if e["event_type"] == "procedure"}
        for r in rules:
            if r["event_type"] == "procedure" and r.get("procedure") not in llm_procs:
                events.append(dict(r, needs_confirmation=False, confirm_reasons=[], source="rules"))
        rule_vit = next((r for r in rules if r["event_type"] == "vital"), None)
        if rule_vit:
            llm_vit = next((e for e in events if e["event_type"] == "vital"), None)
            if llm_vit is None:
                events.append(dict(rule_vit, needs_confirmation=False, confirm_reasons=[], source="rules"))
            else:
                for k, v in rule_vit["vital"].items():
                    llm_vit["vital"].setdefault(k, v)
        if any(e["event_type"] != "note" for e in events):
            events[:] = [e for e in events if e["event_type"] != "note"]
