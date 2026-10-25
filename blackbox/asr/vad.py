"""
Black Box - streaming voice-activity detection (Silero VAD v6, ONNX, CPU, offline).

Uses the silero_vad_v6.onnx that ships inside the faster-whisper package, so there is
nothing extra to download. Feed 16 kHz float32 audio in any block size; get back one
Clip per spoken utterance (a medic's sentence), cut at pauses.
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

import numpy as np

SR = 16000
FRAME = 512          # samples per VAD step = 32 ms
CONTEXT = 64         # Silero v5/v6 needs the previous 64 samples as context


def default_model_path() -> str:
    from faster_whisper.utils import get_assets_path
    return os.path.join(get_assets_path(), "silero_vad_v6.onnx")


class SileroVAD:
    """Frame-by-frame speech probability, carrying the RNN state between calls."""

    def __init__(self, path: str | None = None):
        import onnxruntime
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.sess = onnxruntime.InferenceSession(path or default_model_path(),
                                                 providers=["CPUExecutionProvider"], sess_options=opts)
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.ctx = np.zeros((1, CONTEXT), np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        assert frame.shape == (FRAME,)
        x = np.concatenate([self.ctx, frame[None, :].astype(np.float32)], axis=1)
        out, self.h, self.c = self.sess.run(None, {"input": x, "h": self.h, "c": self.c})
        self.ctx = x[:, -CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0])


@dataclass
class Clip:
    audio: np.ndarray
    start_s: float      # offset from stream start
    end_s: float

    @property
    def duration(self) -> float:
        return len(self.audio) / SR


class Segmenter:
    """
    Hysteresis state machine:
      start a clip when p >= threshold,
      end it after `min_silence_ms` of p < neg_threshold,
      keep `pad_ms` of audio before and after, drop clips shorter than `min_speech_ms`,
      force-cut clips longer than `max_clip_s` (a medic talking non-stop).
    min_silence_ms=700 suits dictation: short pauses at commas stay inside one clip.
    """

    def __init__(self, vad: SileroVAD | None = None, threshold: float = 0.5, neg_threshold: float = 0.35,
                 min_silence_ms: int = 700, min_speech_ms: int = 250, pad_ms: int = 300, max_clip_s: float = 20.0):
        self.vad = vad or SileroVAD()
        self.thr, self.neg = threshold, neg_threshold
        ms = lambda v: max(1, int(v * SR / 1000 / FRAME))  # noqa: E731
        self.min_sil, self.min_speech, self.pad = ms(min_silence_ms), ms(min_speech_ms), ms(pad_ms)
        self.max_frames = int(max_clip_s * SR / FRAME)
        self.reset()

    def reset(self):
        self.vad.reset()
        self._pending = np.zeros(0, np.float32)
        self._pre: deque[np.ndarray] = deque(maxlen=self.pad)
        self._clip: list[np.ndarray] = []
        self._speech_frames = 0
        self._silence = 0
        self._frame_idx = 0
        self._clip_start = 0
        self.active = False

    def feed(self, audio: np.ndarray) -> list[Clip]:
        """Push any amount of audio; returns clips that finished inside it."""
        self._pending = np.concatenate([self._pending, audio.astype(np.float32).reshape(-1)])
        done = []
        while len(self._pending) >= FRAME:
            frame, self._pending = self._pending[:FRAME], self._pending[FRAME:]
            clip = self._step(frame)
            if clip is not None:
                done.append(clip)
        return done

    def flush(self) -> list[Clip]:
        """Call at end of stream to emit an unfinished clip."""
        if self.active and self._speech_frames >= self.min_speech:
            return [self._emit(trim_silence=False)]
        return []

    # -------------------------------------------------------------
    def _step(self, frame: np.ndarray) -> Clip | None:
        p = self.vad(frame)
        self._frame_idx += 1
        if not self.active:
            self._pre.append(frame)
            if p >= self.thr:
                self.active = True
                self._clip = list(self._pre)
                self._clip_start = self._frame_idx - len(self._pre)
                self._speech_frames, self._silence = 1, 0
            return None

        self._clip.append(frame)
        if p >= self.thr:
            self._speech_frames += 1
            self._silence = 0
        elif p < self.neg:
            self._silence += 1
        if len(self._clip) >= self.max_frames:
            return self._emit(trim_silence=False)
        if self._silence >= self.min_sil:
            if self._speech_frames < self.min_speech:   # a cough / click: drop it
                self._reset_clip()
                return None
            return self._emit(trim_silence=True)
        return None

    def _emit(self, trim_silence: bool) -> Clip:
        frames = self._clip
        if trim_silence:  # keep only `pad` frames of the trailing silence
            frames = frames[: len(frames) - self._silence + self.pad]
        audio = np.concatenate(frames)
        start = self._clip_start * FRAME / SR
        clip = Clip(audio, start, start + len(audio) / SR)
        self._reset_clip()
        return clip

    def _reset_clip(self):
        self.active = False
        self._clip = []
        self._pre.clear()
        self._speech_frames = self._silence = 0