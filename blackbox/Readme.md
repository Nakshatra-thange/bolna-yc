# Black Box: the flight recorder for emergency care

**Voice-first documentation for ambulances that works with no signal, understands Hindi + English, catches dangerous drug doses out loud, and makes the record impossible to quietly change afterwards.**

> A paramedic says *"gave adrenaline one milligram IV"* (or *"adrenaline ek milligram IV diya"*).
> Black Box hears it offline, writes a structured clinical event, reads it back for confirmation,
> checks it against protocol, and seals it in a cryptographically signed chain with the original voice clip as evidence.

## Why this matters
- In emergency care, notes are written from memory hours later. When a dose is disputed, nobody can prove what was said or when.
- Ambulances lose signal; cloud-only voice agents fail exactly when they are needed, and patient audio should not leave the vehicle.
- India needs this in Hindi and Hinglish, not just English.
- Speech recognition is never perfect. So Black Box is built around **catching and confirming mistakes**, not pretending they don't happen.

## What it does
| Stage | What happens | Where |
|---|---|---|
| Listen | Offline speech-to-text (faster-whisper) with voice-activity detection and medical-vocabulary biasing | `asr/` |
| Understand | Rules + a small local LLM turn speech into structured events (drug, dose, unit, route, vitals, procedures). **The LLM only proposes; code verifies** that every number and drug really appears in the transcript | `rule_extractor.py`, `extract/` |
| Hindi/Hinglish | Drug names, Hindi number words (*ek sau bees by assi* → 120/80), units, patient context (*wazan pachaas kilo*), confirm/cancel commands | `normalize.py`, `guardian/context.py`, `extract/confirm.py` |
| Confirm | The device speaks the event back; the crew says confirm or cancel | `extract/confirm.py`, `asr/live.py` |
| Guardian | 10 protocol rules (allergy, max dose, repeat too soon, wrong route, ...) speak up as a "second pair of ears"; an override needs a spoken reason | `guardian/`, `protocol.json` |
| Seal | Every utterance, event, confirmation, alert and answer is canonical-JSON → SHA-256 → Ed25519 signed → linked to the previous entry; voice clips are AES-256-GCM encrypted and hash-bound | `audit/` |
| Sync | When signal returns, the chain is pushed; the hospital server verifies it, pins the device key, rejects forks, and returns a signed anchor receipt | `sync/`, `server/` |
| Hand over | Dashboard with live integrity verdict, voice playback, tamper simulation, FHIR R4 export with Provenance signatures, printable handover sheet | `server/` |

## The demo moment
Run the attack simulation: change one dose after the fact. The dashboard turns red and names the exact entry:
*"Entry #15 fails verification — it was changed after the ambulance signed it."* Play the original voice clip, press Restore, green again.

## Results (reproducible)
- Test suite: **169 passed, 2 skipped** with global Miniforge Python 3.12 (`x86_64`); **189 passed** with `.venv-native` Python 3.12 (`arm64`). Both were run with `python -m pytest -q`.
- Guardian: **23/23** scenes, alert precision **1.00**, recall **1.00**, 0 false alarms (`python eval/score_guardian.py`).
- Extraction on 119 gold dictations (30 hard held-out), rules only, no LLM: headline F1 **97.3**; on the hard held-out set event F1 **83.3**.
  Of 87 recorded medications, 12 were wrong, **10 were caught by confirmation**, 2 slipped through. This is the number the LLM + verification layer exists to improve:
  run `python eval/run_extractor.py --extractor hybrid --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf` then `eval/score.py` and read the `WRONG AND UNFLAGGED` line.
- Tamper tests: edit, delete, reorder, backdate, swap audio and truncate are all detected (truncation requires the server anchor).

## Run it
```bash
python -m pip install -r requirements.txt            # includes fastapi, uvicorn, httpx
python -m pytest -q
rm -rf runs server_data && python -m audit.demo      # scripted cardiac-arrest incident
python -m server.app                                 # http://127.0.0.1:8000
python -m sync.client push                           # second terminal
python -m asr.live --mic --model small --extract --speak   # live voice
```
Full test plan (English + Hindi): `TESTING.md`. Original engineering guide: `docs/DEV_GUIDE.md`.

## Design principles
1. **Offline first.** Everything on the ambulance runs locally; the network is only for sync.
2. **The model reads, code decides.** A language model never gets to invent a dose.
3. **Make mistakes visible, not impossible.** Confirmation, Guardian alerts and corrections-as-new-entries; nothing is overwritten.
4. **Evidence over trust.** Signed chain, pinned keys, server anchors, voice clip bound to the hash.
5. **No real patient data.** All data is synthetic; keys, models and databases are git-ignored.

## Honest limitations
- Live Hindi speech recognition with the small offline Whisper models is weak; Hindi is strongest on typed/Romanized text today. Whisper in Hindi mode outputs Devanagari, which the text layers do not read yet.
- Ambiguous Hindi words (*do*, *ek*) are only treated as numbers next to a unit, by design.
- Drug codes (RxNorm/SNOMED) are best-effort for a hackathon and need clinical review before real use.
- The audio key is escrowed to the demo server for playback (a simplification).
- Not a medical device; not clinically validated.

## Roadmap: Indian-language voice + phone agents (for the Bolna Buildathon)
Not built yet. Planned, in order of value:
1. **Sarvam ASR/TTS as an optional online backend** for accurate Hindi/Indic speech, with the current offline stack as automatic fallback.
2. **Bolna phone agent for pre-arrival handover:** when the ambulance syncs, a Bolna voice agent calls the receiving hospital's ER (via Plivo/Exotel), speaks a 20-second handover in Hindi or English from the *verified* record, and answers questions like "what was given and when?" by reading the signed chain.
3. Extend the Guardian protocol and vocabulary to more drugs and regional brand names.

## Repo layout
`asr/ extract/ audit/ guardian/ server/ sync/ eval/` plus `schema.py protocol.json protocol.py normalize.py rule_extractor.py` and the `test_*.py` files. See `docs/DEV_GUIDE.md` for details.
