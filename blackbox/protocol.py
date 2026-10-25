"""
Black Box - protocol loader.

Validates protocol.json on load (a typo in a dose limit is a safety bug),
resolves spoken drug names to canonical ids, and computes the effective
dose limits for a given patient, indication and route.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from schema import PatientContext, Route

UNIT_FACTORS_TO_MG = {"mg": 1.0, "mcg": 0.001, "g": 1000.0}


# ---------------------------------------------------------------- JSON models
class Source(BaseModel):
    id: str
    title: str
    publisher: str
    date: str
    url: str


class AdultDosing(BaseModel):
    dose_min: float = Field(..., gt=0)
    dose_max: float = Field(..., gt=0)
    max_total: Optional[float] = None
    max_doses: Optional[int] = None
    min_repeat_interval_min: Optional[float] = None

    @model_validator(mode="after")
    def ordered(self):
        if self.dose_min > self.dose_max:
            raise ValueError("dose_min > dose_max")
        if self.max_total is not None and self.max_total < self.dose_max:
            raise ValueError("max_total smaller than a single max dose")
        return self


class PediatricDosing(BaseModel):
    per_kg: Optional[float] = None
    fixed_dose: Optional[float] = None
    max_single: float
    max_total: Optional[float] = None
    max_total_per_kg: Optional[float] = None
    max_doses: Optional[int] = None
    min_repeat_interval_min: Optional[float] = None

    @model_validator(mode="after")
    def has_a_dose(self):
        if self.per_kg is None and self.fixed_dose is None:
            raise ValueError("pediatric dosing needs per_kg or fixed_dose")
        return self


class Indication(BaseModel):
    routes: list[Route]
    adult: AdultDosing
    pediatric: Optional[PediatricDosing] = None
    note: str = ""
    route_overrides: dict[str, dict[str, float]] = Field(default_factory=dict)
    source: str


class Contraindication(BaseModel):
    id: str
    type: Literal["allergy", "vital", "flag", "age"]
    text: str
    match: list[str] = Field(default_factory=list)   # allergy
    field: Optional[str] = None                       # vital
    op: Optional[Literal["<", "<=", ">", ">="]] = None
    value: Optional[float] = None
    flag: Optional[str] = None                        # flag


class Concentration(BaseModel):
    label: str
    mg_per_ml: float


class Drug(BaseModel):
    generic_name: str
    aliases: list[str]
    drug_class: str
    controlled: bool
    canonical_unit: Literal["mg", "mcg", "g", "mL"]
    concentrations: list[Concentration] = Field(default_factory=list)
    indications: dict[str, Indication]
    contraindications: list[Contraindication] = Field(default_factory=list)
    dangerous_mismatches: list[dict] = Field(default_factory=list)
    verified_against_source: bool = False


class PediatricRule(BaseModel):
    description: str
    age_lt_years: float
    weight_lt_kg: float


class Protocol(BaseModel):
    protocol_id: str
    version: str
    disclaimer: str
    sources: list[Source]
    pediatric_rule: PediatricRule
    drugs: dict[str, Drug]
    unit_conversions: dict[str, float]

    @model_validator(mode="after")
    def integrity(self):
        source_ids = {s.id for s in self.sources}
        seen: dict[str, str] = {}
        for drug_id, drug in self.drugs.items():
            for ind_id, ind in drug.indications.items():
                if ind.source not in source_ids:
                    raise ValueError(f"{drug_id}.{ind_id}: unknown source '{ind.source}'")
            for alias in drug.aliases:
                key = alias.lower()
                if key in seen and seen[key] != drug_id:
                    raise ValueError(f"alias '{alias}' used by both {seen[key]} and {drug_id}")
                seen[key] = drug_id
        return self


# ---------------------------------------------------------------- helpers
@dataclass(frozen=True)
class DoseLimits:
    drug: str
    indication: str
    route: str
    unit: str
    pediatric: bool
    min_single: Optional[float]
    max_single: float
    max_total: Optional[float]
    max_doses: Optional[int]
    min_repeat_interval_min: Optional[float]
    source: str


class ProtocolBook:
    def __init__(self, protocol: Protocol):
        self.p = protocol
        self._alias = {a.lower(): d for d, drug in protocol.drugs.items() for a in drug.aliases}
        for drug_id in protocol.drugs:
            self._alias.setdefault(drug_id.lower(), drug_id)

    @classmethod
    def load(cls, path: str | Path = Path(__file__).with_name("protocol.json")) -> "ProtocolBook":
        with open(path, encoding="utf-8") as f:
            return cls(Protocol.model_validate(json.load(f)))

    def resolve_drug(self, spoken: str) -> Optional[str]:
        """'Epi' -> 'adrenaline', 'Ondem' -> 'ondansetron'. None if unknown."""
        return self._alias.get(spoken.strip().lower())

    def is_pediatric(self, patient: PatientContext) -> bool:
        rule = self.p.pediatric_rule
        if patient.age_years is not None and patient.age_years < rule.age_lt_years:
            return True
        if patient.weight_kg is not None and patient.weight_kg < rule.weight_lt_kg:
            return True
        return False

    def dose_limits(self, drug_id: str, indication: str, route: str,
                    patient: PatientContext) -> DoseLimits:
        drug = self.p.drugs[drug_id]
        ind = drug.indications[indication]
        if route not in [r.value for r in ind.routes]:
            raise ValueError(f"{route} is not an approved route for {drug_id}/{indication}")
        override = ind.route_overrides.get(route, {})

        if self.is_pediatric(patient) and ind.pediatric is not None:
            ped = ind.pediatric
            per_kg = override.get("pediatric_per_kg", ped.per_kg)
            cap = override.get("pediatric_max_single", ped.max_single)
            if per_kg is not None:
                if patient.weight_kg is None:
                    raise ValueError("pediatric weight-based dosing needs patient weight")
                max_single = min(per_kg * patient.weight_kg, cap)
            else:
                max_single = min(ped.fixed_dose, cap)
            max_total = ped.max_total
            if ped.max_total_per_kg is not None and patient.weight_kg is not None:
                by_kg = ped.max_total_per_kg * patient.weight_kg
                max_total = min(by_kg, max_total) if max_total else by_kg
            return DoseLimits(drug_id, indication, route, drug.canonical_unit, True,
                              None, round(max_single, 4), max_total, ped.max_doses,
                              ped.min_repeat_interval_min, ind.source)

        if self.is_pediatric(patient) and ind.pediatric is None:
            raise ValueError(f"{drug_id}/{indication} has no pediatric dosing in this protocol")

        a = ind.adult
        return DoseLimits(drug_id, indication, route, drug.canonical_unit, False,
                          a.dose_min, override.get("adult_dose_max", a.dose_max),
                          a.max_total, a.max_doses, a.min_repeat_interval_min, ind.source)


def to_canonical(dose: float, unit: str, canonical_unit: str) -> float:
    """Convert a spoken dose into the drug's canonical unit (mg/mcg/g). mL passes through."""
    if unit == canonical_unit:
        return dose
    if unit == "mL" or canonical_unit == "mL":
        raise ValueError("volume <-> mass needs a concentration; resolve it before this step")
    return dose * UNIT_FACTORS_TO_MG[unit] / UNIT_FACTORS_TO_MG[canonical_unit]