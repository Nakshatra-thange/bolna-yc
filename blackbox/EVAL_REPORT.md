# Black Box ASR evaluation

## Current measured result

Primary clean synthetic set: 119 eSpeak clips (`condition=tts`), scored against all 119 gold transcripts. These are synthetic voices, not real ambulance recordings. The small English-prompt JSONL below is an existing saved run; no transcripts or lexicon rules were tuned against `hard_heldout`.

| Config / saved run | Raw WER | Normalized WER | English WER (n=90) | Hinglish WER (n=29) | Hard held-out WER (n=30) | Drug ok (n=84) | Dose ok (n=84) | RTF | p95 latency | Wrong and unflagged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base, prompt, `asr_fw-base_prompt.jsonl` | 65.1% | 48.8% | 43.0% | 65.0% | 53.7% | 59.5% | 81.0% | 0.39 | 2.16 s | not re-extracted |
| small, prompt, English, `asr_fw-small_prompt__tts.jsonl` | 52.4% | 34.3% | 25.1% | 60.3% | 43.8% | 73.8% | 77.4% | 0.68 | 3.42 s | **2/63 (3.2%)** |

Raw WER lowercases and removes punctuation (preserving a decimal point inside a numeric token); it does not equate spoken numbers with digits or expand abbreviations. Normalized WER also converts English/Hinglish number words with `normalize.words_to_numbers`, folds unit and route spellings, and removes punctuation. Both are pooled word error rates, and both are reported so normalization does not hide lexical errors. Drug and dose accuracy are separate medication entity recall measures (n differs by entity availability).

The prior report quoted base WER 54.4% (n=119) and small WER 40.0% (n=119). Those headline values were produced by an earlier evaluation state and do not reproduce from the saved transcript artifacts listed above. On the current saved artifacts, base-to-small moved normalized WER from 48.8% (n=119) to 34.3% (n=119; 14.5 points); raw WER also improved, from 65.1% (n=119) to 52.4% (n=119; 12.7 points). This is a comparison of the saved runs, not a controlled same-process model benchmark. The small run improves drug recall substantially over base but dose recall is slightly lower.

**The <5% target was not reached.** The best measured normalized English WER here is 25.1% (n=90); Hinglish remains 60.3% (n=29). The hard-held-out slice is listed separately in the results table and was not used for prompt or lexicon tuning. There is no Romanized-Hindi normalization in the score yet; the Hinglish result should be read as a known limitation, not mixed into the English result.

## Interview cheat sheet: what we evaluated

| Evaluation | What it answers in everyday language | Result to quote |
|---|---|---|
| **Speech recognition: raw WER** | How many words differ if we only lowercase and remove punctuation? | Small prompt run: **52.4% (119 clips)**. This keeps number and abbreviation mismatches visible. |
| **Speech recognition: normalized WER** | How many words differ after treating equivalent numbers, units and routes alike? | **34.3% (119 clips)** overall; **25.1% (90 English clips)**; **60.3% (29 Hinglish clips)**. The hard-held-out result is in the main table above; its tag can overlap with language tags. |
| **Character error rate (CER)** | How different are the characters, which helps diagnose spelling and sound-alike errors? | **18.1% (119 clips)** on the saved small-prompt artifact. CER is a different metric from WER and can look better when words are still wrong. |
| **Medication entity recall** | Did the transcript contain each gold drug and dose? | Drug: **73.8% (84 medication mentions)**; dose: **77.4% (84 mentions)**. These are recall checks, not proof that the full medication instruction is safe. |
| **Speed** | Is decoding faster than the audio, and how long do individual clips take? | RTF **0.68 (119 clips)** and p95 **3.42 s (119 clips)** on the selected small-prompt run. RTF below 1 means faster than real time. |
| **Text-to-events extraction** | If the extractor receives the perfect written gold transcript, can it produce structured events? | Rules extractor: headline F1 **97.3% (119 cases)** and exact-match **88.2% (119 cases)**. This isolates extraction from speech recognition. |
| **Speech-to-events end-to-end** | What happens when ASR output is actually sent into event extraction? | Headline F1 **72.2% (119 cases)**, event-detection F1 **67.2% (119 cases)**, exact-match **45.4% (119 cases)**. This is the selected small-prompt run. |
| **Medication safety check** | How often did a wrong extracted medication avoid the confirmation flag? | **2 wrong-and-unflagged out of 63 recorded medications (3.2%)** on that same end-to-end run. The goal is to reduce this without increasing unflagged errors. |
| **Event-intent check** | Did the system identify the kind of event in each gold transcript? | **95.8% (119 cases)** exact event-type-set accuracy on gold text. |
| **Spoken command check** | Does the small confirm/cancel parser understand a short hand-written set of commands? | **86.8% (38 examples)** accuracy; **0 wrong confirms and 0 wrong cancels (38 examples)**. This small hand-written check is not a field benchmark. |
| **Noise conditions** | How does recognition behave in the saved road/siren audio variants? | Archived small-prompt run, **9 clips per condition**: clean 23.5%, road 0 dB 36.8%, road 5 dB 41.2%, road 10 dB 32.4%, siren 0 dB 29.4%, siren 5 dB 30.9%, siren 10 dB 32.4% normalized WER. These slices are too small to draw strong conclusions. |
| **Automated tests** | Do the existing code checks still pass? | `pytest -q`: **179 passed, 2 skipped**. Passing tests do not measure speech accuracy or clinical readiness. |

**A concise way to present it:** “We evaluate each layer separately. With perfect text, rules-based extraction scores 97.3% headline F1 (119 cases), but after speech recognition it is 72.2% (119 cases). The selected small Whisper run has 25.1% normalized WER on English (90 clips), 60.3% on Hinglish (29), and a 2/63 wrong-and-unflagged medication count. The audio is synthetic, so these are engineering results, not evidence of performance with ambulance crews.”

## How to make the evaluation better organically

These improve the evidence and can also reveal real model improvements; they do not make WER look better by changing the scoring rules.

1. **Record more natural, consented speech.** Use volunteers reading the existing script plus spontaneous paraphrases, with varied accents, speaking rates, microphones and realistic background noise. Keep every recording synthetic-free and patient-free, and report real speech separately from eSpeak/Piper.
2. **Make references more trustworthy.** Have two people independently transcribe each recording, then resolve disagreements. Preserve the verbatim transcript for raw WER; separately annotate accepted number/unit/route variants for normalized scoring.
3. **Prevent leakage.** Split by speaker and base utterance before making noisy copies. Keep every version of the same utterance in one split. Tune prompts and lexicon only on development speakers; run the held-out set once for the final comparison.
4. **Balance the test set.** Add enough English and code-switched Hinglish from each speaker, and include ordinary speech, medication names, doses, routes, corrections, negation, and no-medication statements. Publish the counts for each slice.
5. **Measure uncertainty, not just one percentage.** Report reference word counts, sample counts, and bootstrap confidence intervals. Keep raw and normalized WER side by side, and report drug, dose, unit and route separately, including false drug insertions as well as misses.
6. **Make model comparisons controlled.** Run each model and decode setting on exactly the same clips and machine, record model/library versions and command line, and report RTF plus p50/p95 latency. Do not compare latency from different saved runs as if they were measured in one session.
7. **Use targeted error review on development data.** Classify errors (number, drug, route, language switching, noise, hallucination), fix the underlying issue, and then check the untouched held-out set. Do not add rules that map arbitrary speech to a drug name or weaken the protected-word safeguards.

The highest-value next evaluation step is a speaker-separated, naturally recorded volunteer set with independently checked transcripts. It will tell us whether the current eSpeak results predict real speech performance; Piper alone would still be synthetic evidence.

## Decode and audio coverage

| Attempt | Result |
|---|---|
| Small, English, beam 5, prompt, independent clips | Existing saved run measured above. RTF 0.68 (<1); p95 3.42 s. |
| Small without bias / hotwords | Existing saved tts predictions score 45.3% / 52.8% normalized WER respectively. They are not part of the controlled table because their saved latency profiles are inconsistent with the selected small prompt run. |
| Medium, distil-large-v3, large-v3; fresh English/Hinglish auto-vs-hi runs | Not run. Constructing a local faster-whisper model fails in this environment: installed NumPy is 2.3.4 while compiled Torch/CTranslate2 components report NumPy 1.x ABI incompatibility; construction then exits on an OpenMP shared-memory error. No packages or models were installed to alter the environment. |
| Natural offline TTS (Piper) | Not run; `piper` is not installed. |
| Real voice recordings | None in the evaluation data. `eval/record.py` is present and records the scripted lines to separate condition folders. Keep those scores separate from synthetic audio. Do not record patient data. |

Existing audio contains eSpeak `tts` and noisy variants (`clean`, road/siren at 0/5/10 dB), not a Piper-natural set. The noisy variants are reported by their own condition in `voice_report.py`; they are not mixed into the 119-clip primary TTS score. No real-voice or natural-TTS WER is claimed.

## Safety and tests

The selected small prompt output was passed through rules extraction end-to-end. It produced **2 wrong-and-unflagged medication events out of 63 recorded medications**, equal to the supplied best count of 2/63; the count did not increase. The scorer reports 2/60 for the other all-condition small-prompt artifact, so the denominator is tied to the exact prediction set. Safety counts are not interchangeable across those prediction files.

The existing end-to-end summary used gold text and reported 2/87. That is a different measurement from ASR end-to-end and must not be conflated with 2/63 above.

Tests: `pytest -q` → **179 passed, 2 skipped**. The new normalizer tests cover literal-vs-normalized scoring, decimal and colloquial numbers, unit/route forms, and preservation of drug identity.

## Reproduction

Score the saved ASR run with the revised raw and normalized metrics:

```bash
python eval/voice_report.py --asr eval/preds/asr_fw-small_prompt__tts.jsonl
```

Create a fresh English small-model run where the local ASR runtime is working (beam size 5, prompt bias, no previous-text carryover in the engine):

```bash
python eval/run_asr.py --backend faster-whisper --model small --bias prompt --language en --beam-size 5 --conditions tts --out eval/preds/asr_repro_small_en.jsonl
python eval/voice_report.py --asr eval/preds/asr_repro_small_en.jsonl
```

Hinglish language comparison, when model construction is available:

```bash
python eval/run_asr.py --model small --bias prompt --language hi --conditions tts --out eval/preds/asr_repro_small_hi.jsonl
python eval/run_asr.py --model small --bias prompt --language auto --conditions tts --out eval/preds/asr_repro_small_auto.jsonl
```

Reproduce the selected small run's safety number:

```bash
python eval/run_extractor.py --extractor rules --input eval/preds/asr_fw-small_prompt__tts.jsonl --tag e2e_best
python eval/score.py --pred eval/preds/ext_e2e_best.jsonl --label e2e_best
python eval/voice_report.py --asr eval/preds/asr_fw-small_prompt__tts.jsonl
```

The extraction command intentionally uses the tts-only ASR JSONL without `--condition`, so it writes `ext_e2e_best.jsonl` as shown. The scorer uses all available gold rows and does not omit hard clips. The WER slices are descriptive; only the English development subset may be used for future prompt or lexicon tuning. Keep `hard_heldout` untouched for evaluation.
