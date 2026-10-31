
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
AUDIO = ROOT / "eval" / "data" / "audio"
SR = 16000


def synth_siren(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """'Wail' siren: tone sweeping 650-1500 Hz every ~4 s, with harmonics and a little wobble."""
    t = np.arange(n) / sr
    period = 4.0 + rng.uniform(-0.5, 0.5)
    lfo = 0.5 * (1 - np.cos(2 * np.pi * t / period))           # 0..1..0
    freq = 650 + 850 * lfo
    phase = 2 * np.pi * np.cumsum(freq) / sr + rng.uniform(0, 2 * np.pi)
    s = np.sin(phase) + 0.35 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
    s *= 0.8 + 0.2 * np.sin(2 * np.pi * 0.3 * t)               # distance / doppler-ish level change
    return s.astype(np.float32)


def synth_road(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Road + engine: brown noise rumble, 2-cylinder-ish hum harmonics, occasional bumps."""
    brown = np.cumsum(rng.standard_normal(n))
    brown -= np.convolve(brown, np.ones(sr // 20) / (sr // 20), mode="same")   # remove DC drift
    brown /= np.abs(brown).max() + 1e-9
    t = np.arange(n) / sr
    rpm_hz = 38 + 6 * np.sin(2 * np.pi * 0.1 * t)
    ph = 2 * np.pi * np.cumsum(rpm_hz) / sr
    hum = sum(np.sin(k * ph) / k for k in range(1, 6))
    bumps = np.zeros(n)
    for start in rng.integers(0, max(n - sr // 10, 1), size=max(1, n // (sr * 3))):
        L = sr // 10
        bumps[start:start + L] += np.hanning(L)[: n - start] * rng.standard_normal(min(L, n - start))
    hiss = 0.05 * rng.standard_normal(n)
    return (brown + 0.4 * hum / 2 + 0.6 * bumps + hiss).astype(np.float32)


def load_noise(path: Path, n: int, rng: np.random.Generator) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:  # simple linear resample, fine for noise
        x = np.interp(np.arange(0, len(x), sr / SR), np.arange(len(x)), x).astype(np.float32)
    if len(x) < n:
        x = np.tile(x, n // len(x) + 1)
    start = rng.integers(0, len(x) - n + 1)
    return x[start:start + n]


def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    # speech power from the louder half of frames, so silence at the ends doesn't skew SNR
    frames = speech[: len(speech) // 400 * 400].reshape(-1, 400)
    e = (frames ** 2).mean(axis=1)
    p_speech = e[e >= np.median(e)].mean() if len(e) else (speech ** 2).mean()
    p_noise = (noise ** 2).mean() + 1e-12
    noise = noise * np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    out = speech + noise
    peak = np.abs(out).max()
    return out / peak * 0.95 if peak > 0.95 else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, nargs="+", default=[10, 5, 0])
    ap.add_argument("--siren-file", type=Path)
    ap.add_argument("--road-file", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    clean = sorted((AUDIO / "clean").glob("*.wav"))
    if not clean:
        raise SystemExit(f"no recordings in {AUDIO / 'clean'} - run eval/record.py first")
    for wav in clean:
        x, sr = sf.read(wav, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            raise SystemExit(f"{wav.name} is {sr} Hz, expected {SR}")
        for kind in ("siren", "road"):
            f = a.siren_file if kind == "siren" else a.road_file
            noise = load_noise(f, len(x), rng) if f else (synth_siren if kind == "siren" else synth_road)(len(x), SR, rng)
            for snr in a.snr:
                d = AUDIO / f"{kind}_{snr:g}db"
                d.mkdir(parents=True, exist_ok=True)
                sf.write(d / wav.name, mix(x, noise, snr), SR, subtype="PCM_16")
    print(f"made {len(clean) * 2 * len(a.snr)} noisy files from {len(clean)} clean recordings in {AUDIO}")


if __name__ == "__main__":
    main()