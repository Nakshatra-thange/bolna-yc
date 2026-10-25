"""
Black Box - offline ASR engines behind one interface.

  faster-whisper  (default, laptop/edge box)  supports vocabulary biasing: initial_prompt and/or hotwords
  sherpa-whisper  (sherpa-onnx, same engine family you'd ship on Android)  no prompt biasing -> rely on lexicon fix

Both return a Transcript. Nothing here touches the network once models are on disk.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from asr.vocab import hotwords, initial_prompt

SR = 16000
Bias = Literal["none", "prompt", "hotwords", "both"]


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float


@dataclass
class Transcript:
    text: str
    words: list[Word] = field(default_factory=list)
    avg_logprob: Optional[float] = None    # whisper decoder confidence (closer to 0 = surer)
    no_speech_prob: Optional[float] = None
    latency_s: float = 0.0
    audio_s: float = 0.0
    engine: str = ""

    @property
    def confidence(self) -> float:
        """0..1 utterance confidence. Mean word prob if we have it, else exp(avg_logprob)."""
        if self.words:
            return float(np.mean([w.prob for w in self.words]))
        if self.avg_logprob is not None:
            return float(np.exp(self.avg_logprob))
        return 0.5


class FasterWhisperEngine:
    def __init__(self, model: str = "small", device: str = "cpu", compute_type: str = "int8",
                 bias: Bias = "prompt", language: Optional[str] = "en", beam_size: int = 5,
                 word_timestamps: bool = True):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.bias, self.language, self.beam = bias, language, beam_size
        self.word_ts = word_timestamps
        self.name = f"faster-whisper:{model}:{bias}"

    def transcribe(self, audio: np.ndarray) -> Transcript:
        t0 = time.perf_counter()
        segs, _info = self.model.transcribe(
            audio.astype(np.float32),
            language=self.language,
            beam_size=self.beam,
            initial_prompt=initial_prompt() if self.bias in ("prompt", "both") else None,
            hotwords=hotwords() if self.bias in ("hotwords", "both") else None,
            word_timestamps=self.word_ts,
            condition_on_previous_text=False,   # each clip independent: no hallucination carry-over
            vad_filter=False,                   # we already cut clips with our own VAD
            temperature=0.0,                    # deterministic, no sampling fallback surprises
        )
        segs = list(segs)
        words = [Word(w.word.strip(), w.start, w.end, w.probability) for s in segs for w in (s.words or [])]
        return Transcript(
            text=" ".join(s.text.strip() for s in segs).strip(),
            words=words,
            avg_logprob=float(np.mean([s.avg_logprob for s in segs])) if segs else None,
            no_speech_prob=float(np.max([s.no_speech_prob for s in segs])) if segs else None,
            latency_s=time.perf_counter() - t0,
            audio_s=len(audio) / SR,
            engine=self.name,
        )


class SherpaWhisperEngine:
    """sherpa-onnx Whisper. Download e.g.
    https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-small.en.tar.bz2"""

    def __init__(self, model_dir: str, threads: int = 2, int8: bool = True):
        import sherpa_onnx
        d = Path(model_dir)
        prefix = d.name.replace("sherpa-onnx-whisper-", "")
        q = ".int8" if int8 else ""
        self.rec = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(d / f"{prefix}-encoder{q}.onnx"),
            decoder=str(d / f"{prefix}-decoder{q}.onnx"),
            tokens=str(d / f"{prefix}-tokens.txt"),
            num_threads=threads,
        )
        self.name = f"sherpa-whisper:{prefix}"

    def transcribe(self, audio: np.ndarray) -> Transcript:
        t0 = time.perf_counter()
        s = self.rec.create_stream()
        s.accept_waveform(SR, audio.astype(np.float32))
        self.rec.decode_stream(s)
        return Transcript(text=s.result.text.strip(), latency_s=time.perf_counter() - t0,
                          audio_s=len(audio) / SR, engine=self.name)


def make_engine(backend: str = "faster-whisper", **kw):
    if backend == "faster-whisper":
        return FasterWhisperEngine(**{k: v for k, v in kw.items() if k in
                                      ("model", "device", "compute_type", "bias", "language", "beam_size")})
    if backend == "sherpa-whisper":
        return SherpaWhisperEngine(kw["model_dir"])
    raise ValueError(f"unknown backend {backend}")