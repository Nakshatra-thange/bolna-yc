"""Tests for the Phase 2 speech layer.  python -m pytest -q
VAD tests synthesize speech with offline eSpeak (pip install espeakng-loader); skipped if missing."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from asr.engine import Transcript
from asr.vocab import LexiconCorrector, hotwords, initial_prompt, phonetic

ROOT = Path(__file__).parent
FIX = LexiconCorrector()


# ------------------------------------------------------------------ vocabulary biasing
def test_prompt_is_short_and_has_drugs():
    p = initial_prompt()
    assert len(p.split()) < 120          # Whisper prompt budget is ~224 tokens
    for w in ("epinephrine", "amiodarone", "glucagon", "Ondem", "IV"):
        assert w in p


def test_hotwords_skip_risky_short_aliases():
    hw = hotwords().split()
    assert "glucagon" in hw and "ms" not in hw and "ns" not in hw


def test_phonetic_sound_alikes():
    assert phonetic("morfeen") == phonetic("morphine")
    assert phonetic("versed")[:3] == phonetic("first")[:3]


# ------------------------------------------------------------------ lexicon correction
@pytest.mark.parametrize("heard,expected", [
    ("administered blue cannon 1 millid ram", "milligram"),
    ("Game 500 milliliters on sailing", "saline"),
    ("Okay now look so near 0.4 mg", "naloxone"),
    ("Game 5 and Morphin Roman 4", "morphine IV"),
    ("pushing for milligrams morphine", "4 milligrams"),
    ("0.3 milligram mintram muscular", "intramuscular"),
])
def test_corrector_fixes(heard, expected):
    fixed, fixes = FIX.correct(heard)
    assert expected in fixed and fixes


@pytest.mark.parametrize("text", [
    "line laga di, 18 gauge",
    "NS 500cc bolus wide open",
    "pt c/o 8/10 chest pain, gave ASA 324 chewed",
    "patient ko narcan 0.4 milligrams line mein diya",
    "I'm out of the room, back in 5 minutes",
])
def test_corrector_never_invents_drugs(text):
    fixed, fixes = FIX.correct(text)
    assert fixed == text and fixes == []


def test_corrector_leaves_all_gold_transcripts_alone():
    """On perfect transcripts the only allowed change is cosmetic 'I V' -> 'IV'."""
    for line in open(ROOT / "eval" / "data" / "gold.jsonl", encoding="utf-8"):
        g = json.loads(line)
        _, fixes = FIX.correct(g["text"])
        assert all(f.kind == "known" and f.fixed in ("IV", "IM", "IO") for f in fixes), (g["id"], fixes)


# ------------------------------------------------------------------ VAD + pipeline (no ASR model needed)
espeak = pytest.importorskip("espeakng_loader")


@pytest.fixture(scope="module")
def tts():
    import sys
    sys.path.insert(0, str(ROOT / "eval"))
    from synth_tts import ESpeak
    return ESpeak()


def test_vad_cuts_one_clip_per_sentence(tts):
    from asr.vad import Segmenter
    gap = np.zeros(int(1.5 * 16000), np.float32)
    lines = ["epi one milligram IV", "amiodarone three hundred milligrams", "BP one twenty over eighty"]
    audio = np.concatenate(sum([[tts.say(t), gap] for t in lines], []))
    seg, clips = Segmenter(), []
    for i in range(0, len(audio), 1000):            # odd block size, like a mic
        clips += seg.feed(audio[i:i + 1000])
    clips += seg.flush()
    assert len(clips) == 3
    assert all(0.5 < c.duration < 6 for c in clips)


def test_vad_ignores_silence_and_noise():
    from asr.vad import Segmenter
    noise = (np.random.default_rng(0).standard_normal(16000 * 3) * 0.003).astype(np.float32)
    seg = Segmenter()
    assert seg.feed(noise) + seg.flush() == []


def test_mic_capture_is_paused_during_spoken_prompt():
    import queue
    from asr.live import MicCaptureGate

    q = queue.Queue()
    gate = MicCaptureGate(q)
    block = np.ones((1600, 1), np.float32)
    gate.callback(block)
    assert q.qsize() == 1

    gate.during_playback(lambda _: gate.callback(block), "confirmation prompt")
    assert q.qsize() == 1
    gate.callback(block)
    assert q.qsize() == 2


class FakeEngine:
    name = "fake"

    def transcribe(self, audio):
        return Transcript(text="pushing for milligrams morfeen Roman 4", audio_s=len(audio) / 16000, engine="fake")


def test_pipeline_logs_raw_fixed_hash_and_events(tts, tmp_path):
    from asr.live import Pipeline
    from asr.vad import Clip
    pipe = Pipeline(FakeEngine(), "INC-T", tmp_path, extract=True, keys_dir=tmp_path / "keys")
    rec = pipe.handle(Clip(tts.say("morphine four milligrams IV"), 0.0, 2.0), datetime.now(timezone.utc))
    assert rec["raw_text"].startswith("pushing for") and "4 milligrams morphine IV" in rec["text"]
    assert len(rec["audio_sha256"]) == 64 and (tmp_path / rec["clip_path"]).exists()
    med = [e for e in rec["events"] if e["event_type"] == "medication"][0]
    assert (med["drug"], med["dose"], med["route"]) == ("morphine", 4, "IV")
