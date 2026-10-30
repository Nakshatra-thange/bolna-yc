"""Hinglish regression tests: Hindi number words, doses, vitals, patient context."""
from datetime import datetime, timezone

import pytest

from guardian.context import parse_context
from normalize import words_to_numbers
from rule_extractor import extract

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("spoken,expected", [
    ("ek sau bees by assi", "120 by 80"),
    ("do milligram morphine", "2 milligram morphine"),
    ("pachaas mcg fentanyl", "50 mcg fentanyl"),
    ("dedh sau mg", "150 mg"),
    ("saade teen mg", "3.5 mg"),
    ("teen sau joule", "300 joule"),
    ("do sau pachaas ml", "250 ml"),
])
def test_hindi_numbers_convert(spoken, expected):
    assert words_to_numbers(spoken) == expected


@pytest.mark.parametrize("spoken", ["aspirin do", "adrenaline do", "ek baar"])
def test_ambiguous_hindi_words_are_not_turned_into_doses(spoken):
    assert not any(c.isdigit() for c in words_to_numbers(spoken))


def test_hinglish_dose_extracted():
    ev = extract("morphine do milligram diya patient ko", NOW)[0]
    assert (ev["drug"], ev["dose"], ev["unit"]) == ("morphine", 2.0, "mg")


def test_hinglish_bp_extracted():
    ev = extract("BP ek sau bees by assi", NOW)[0]
    assert ev["event_type"] == "vital" and ev["vital"] == {"sbp": 120, "dbp": 80}


def test_hinglish_patient_context():
    assert parse_context("wazan pachaas kilo, umar das saal") == {"weight_kg": 50.0, "age_years": 10.0}
    assert parse_context("patient ko penicillin ki allergy hai")["allergies"] == ["penicillin"]