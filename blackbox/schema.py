"""
Black Box - event schema (v1).

Every utterance a medic speaks becomes one ClinicalEvent. Later (Phase 3),
each event is canonicalised, hashed and signed into the audit chain, so
everything here is designed to be deterministic and serialisable.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"
CONFIRM_THRESHOLD = 0.85  # below this, the medic must say "confirm"


# ---------------------------------------------------------------- vocabularies
class EventType(str, Enum):
    MEDICATION = "medication"
    PROCEDURE = "procedure"
    VITAL = "vital"
    NOTE = "note"
    # audit-chain event types (Guardian + corrections live in the same chain)
    ALERT = "alert"
    ALERT_RESPONSE = "alert_response"
    CORRECTION = "correction"


class Route(str, Enum):
    IV = "IV"
    IO = "IO"
    IM = "IM"
    IN = "IN"      # intranasal
    PO = "PO"      # oral / chewed
    SL = "SL"      # sublingual
    NEB = "NEB"    # nebulised
    SC = "SC"
    ODT = "ODT"    # orally disintegrating tablet
    ET = "ET"


class Unit(str, Enum):
    MG = "mg"
    MCG = "mcg"
    G = "g"
    ML = "mL"
    UNITS = "units"


class SpeakerRole(str, Enum):
    PARAMEDIC = "paramedic"
    EMT = "emt"
    DRIVER = "driver"
    PHYSICIAN = "physician"
    UNKNOWN = "unknown"


class TimeSource(str, Enum):
    LOGGED = "logged"        # no time spoken -> time_said = time_logged
    EXPLICIT = "explicit"    # "given at 14:32"
    RELATIVE = "relative"    # "five minutes ago" -> resolved by normaliser


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertAction(str, Enum):
    ACKNOWLEDGED = "acknowledged"
    OVERRIDDEN = "overridden"
    CANCELLED_DOSE = "cancelled_dose"


# ---------------------------------------------------------------- sub-objects
class Speaker(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str = Field(..., description="Crew member id, e.g. 'P1'")
    role: SpeakerRole = SpeakerRole.UNKNOWN


class MedicationDetails(BaseModel):
    drug: str = Field(..., description="Canonical generic id from protocol.json, e.g. 'adrenaline'")
    drug_as_spoken: Optional[str] = Field(None, description="Raw mention, e.g. 'epi'")
    dose: float = Field(..., gt=0)
    unit: Unit
    route: Route
    indication: Optional[str] = Field(None, description="Protocol indication key, e.g. 'cardiac_arrest'")
    concentration: Optional[str] = Field(None, description="e.g. '0.1 mg/mL'")


class ProcedureDetails(BaseModel):
    name: str = Field(..., description="e.g. 'defibrillation', 'iv_access', 'intubation'")
    site: Optional[str] = None
    attempts: Optional[int] = Field(None, ge=1)
    success: Optional[bool] = None
    parameters: dict[str, float | str] = Field(default_factory=dict, description="e.g. {'joules': 200}")


class VitalDetails(BaseModel):
    hr: Optional[int] = Field(None, ge=0, le=350)
    sbp: Optional[int] = Field(None, ge=0, le=350)
    dbp: Optional[int] = Field(None, ge=0, le=250)
    rr: Optional[int] = Field(None, ge=0, le=100)
    spo2: Optional[int] = Field(None, ge=0, le=100)
    gcs: Optional[int] = Field(None, ge=3, le=15)
    bgl_mg_dl: Optional[float] = Field(None, ge=0, le=2000)
    temp_c: Optional[float] = Field(None, ge=20, le=45)

    @model_validator(mode="after")
    def at_least_one(self):
        if all(v is None for v in self.model_dump().values()):
            raise ValueError("vital event needs at least one value")
        return self


class AlertDetails(BaseModel):
    rule_id: str = Field(..., description="e.g. 'MIN_REPEAT_INTERVAL'")
    severity: AlertSeverity
    message: str
    triggered_by_event_id: str


class AlertResponseDetails(BaseModel):
    alert_event_id: str
    action: AlertAction
    reason: Optional[str] = Field(None, description="Spoken reason, required for overrides")

    @model_validator(mode="after")
    def override_needs_reason(self):
        if self.action == AlertAction.OVERRIDDEN and not self.reason:
            raise ValueError("an override must carry a spoken reason")
        return self


class CorrectionDetails(BaseModel):
    corrects_event_id: str
    reason: str
    corrected_fields: dict[str, object] = Field(..., min_length=1)


class PatientContext(BaseModel):
    """Live patient state the Guardian checks against."""
    weight_kg: Optional[float] = Field(None, gt=0, le=400)
    age_years: Optional[float] = Field(None, ge=0, le=130)
    allergies: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list, description="e.g. ['pde5_inhibitor_48h', 'pregnant']")


# ---------------------------------------------------------------- the event
_DETAIL_FIELD = {
    EventType.MEDICATION: "medication",
    EventType.PROCEDURE: "procedure",
    EventType.VITAL: "vital",
    EventType.ALERT: "alert",
    EventType.ALERT_RESPONSE: "alert_response",
    EventType.CORRECTION: "correction",
    EventType.NOTE: None,
}


class ClinicalEvent(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    schema_version: str = SCHEMA_VERSION
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    incident_id: str
    event_type: EventType

    # time
    time_said: datetime = Field(..., description="When the action actually happened")
    time_logged: datetime = Field(..., description="When the device captured the utterance")
    time_source: TimeSource = TimeSource.LOGGED

    # who / evidence
    speaker: Speaker
    transcript: Optional[str] = Field(None, description="Raw ASR text this event came from")
    audio_sha256: Optional[str] = Field(None, pattern=r"^[0-9a-f]{64}$")

    # extraction quality
    confidence: float = Field(..., ge=0.0, le=1.0)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    confirmed: bool = False

    # exactly one of these is filled, matching event_type
    medication: Optional[MedicationDetails] = None
    procedure: Optional[ProcedureDetails] = None
    vital: Optional[VitalDetails] = None
    alert: Optional[AlertDetails] = None
    alert_response: Optional[AlertResponseDetails] = None
    correction: Optional[CorrectionDetails] = None
    note: Optional[str] = None

    @field_validator("time_said", "time_logged")
    @classmethod
    def must_be_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware (use UTC)")
        return v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def check_consistency(self):
        et = EventType(self.event_type)
        required = _DETAIL_FIELD[et]
        details = [f for f in _DETAIL_FIELD.values() if f]
        filled = [f for f in details if getattr(self, f) is not None]

        if et == EventType.NOTE:
            if not self.note:
                raise ValueError("note event needs 'note' text")
            if filled:
                raise ValueError(f"note event must not carry {filled}")
        elif filled != [required]:
            raise ValueError(f"{et.value} event needs exactly '{required}', got {filled or 'none'}")

        if self.time_said > self.time_logged:
            raise ValueError("time_said cannot be after time_logged")
        if TimeSource(self.time_source) == TimeSource.LOGGED and self.time_said != self.time_logged:
            raise ValueError("time_source=logged requires time_said == time_logged")
        if et in (EventType.MEDICATION, EventType.PROCEDURE) and self.audio_sha256 is None:
            raise ValueError("medication/procedure events must link to their audio clip hash")
        return self

    @property
    def needs_confirmation(self) -> bool:
        return not self.confirmed and self.confidence < CONFIRM_THRESHOLD

    def canonical_bytes(self) -> bytes:
        """Deterministic bytes for hashing/signing (sorted keys, no whitespace, no nulls)."""
        data = self.model_dump(mode="json", exclude_none=True)
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")