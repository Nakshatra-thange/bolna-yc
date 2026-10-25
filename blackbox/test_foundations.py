
import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from protocol import ProtocolBook, to_canonical
from schema import ClinicalEvent, PatientContext

NOW = datetime(2026, 10, 22, 10, 30, tzinfo=timezone.utc)
AUDIO = hashlib.sha256(b"fake-audio-clip").hexdigest()
BOOK = ProtocolBook.load()


def med_event(**over):
    base = dict(
        incident_id="INC-001", event_type="medication",
        time_said=NOW, time_logged=NOW, speaker={"id": "P1", "role": "paramedic"},
        transcript="epi one milligram IV given", audio_sha256=AUDIO, confidence=0.93,
        medication={"drug": "adrenaline", "drug_as_spoken": "epi", "dose": 1,
                    "unit": "mg", "route": "IV", "indication": "cardiac_arrest"},
    )
    base.update(over)
    return ClinicalEvent(**base)


# ------------------------------------------------------------- schema
def test_valid_medication_event():
    e = med_event()
    assert e.medication.drug == "adrenaline"
    assert not e.needs_confirmation


def test_low_confidence_needs_confirmation():
    assert med_event(confidence=0.6).needs_confirmation


def test_relative_time_is_allowed():
    e = med_event(time_said=NOW - timedelta(minutes=5), time_source="relative")
    assert e.time_said < e.time_logged


def test_wrong_detail_block_rejected():
    with pytest.raises(ValidationError):
        med_event(event_type="vital")


def test_medication_without_audio_rejected():
    with pytest.raises(ValidationError):
        med_event(audio_sha256=None)


def test_naive_timestamp_rejected():
    with pytest.raises(ValidationError):
        med_event(time_said=NOW.replace(tzinfo=None))


def test_override_needs_reason():
    with pytest.raises(ValidationError):
        ClinicalEvent(incident_id="INC-001", event_type="alert_response",
                      time_said=NOW, time_logged=NOW, speaker={"id": "P1"}, confidence=1.0,
                      alert_response={"alert_event_id": "x", "action": "overridden"})


def test_canonical_bytes_are_deterministic():
    a = med_event(event_id="fixed")
    b = med_event(event_id="fixed")
    assert a.canonical_bytes() == b.canonical_bytes()
    assert b'"drug":"adrenaline"' in a.canonical_bytes()


# ------------------------------------------------------------- protocol
def test_protocol_loads_with_13_drugs():
    assert len(BOOK.p.drugs) == 13


def test_alias_resolution():
    assert BOOK.resolve_drug("Epi") == "adrenaline"
    assert BOOK.resolve_drug("ondem") == "ondansetron"
    assert BOOK.resolve_drug("albuterol") == "salbutamol"
    assert BOOK.resolve_drug("unobtanium") is None


def test_adult_limits():
    lim = BOOK.dose_limits("amiodarone", "cardiac_arrest", "IV", PatientContext(weight_kg=80, age_years=55))
    assert (lim.max_single, lim.max_total, lim.pediatric) == (300, 450, False)


def test_pediatric_weight_based_capped():
    child = PatientContext(weight_kg=20, age_years=6)
    lim = BOOK.dose_limits("adrenaline", "cardiac_arrest", "IV", child)
    assert lim.pediatric and lim.max_single == pytest.approx(0.2)
    lim = BOOK.dose_limits("fentanyl", "pain", "IN", child)
    assert lim.max_single == 20 and lim.max_total == 60


def test_route_override_for_iv_midazolam():
    adult = PatientContext(weight_kg=70, age_years=30)
    assert BOOK.dose_limits("midazolam", "seizure", "IM", adult).max_single == 10
    assert BOOK.dose_limits("midazolam", "seizure", "IV", adult).max_single == 5


def test_unapproved_route_rejected():
    with pytest.raises(ValueError):
        BOOK.dose_limits("aspirin", "acute_coronary_syndrome", "IV", PatientContext(age_years=60))


def test_aspirin_has_no_pediatric_dose():
    with pytest.raises(ValueError):
        BOOK.dose_limits("aspirin", "acute_coronary_syndrome", "PO", PatientContext(age_years=8, weight_kg=25))


def test_unit_conversion():
    assert to_canonical(0.1, "mg", "mcg") == pytest.approx(100)
    assert to_canonical(25000, "mg", "g") == pytest.approx(25)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))