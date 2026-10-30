# Black Box — the flight recorder for emergency care

> Paramedics speak. Black Box turns their words into a structured, timestamped record of every drug and
> procedure — **fully offline** — checks it, reads doubtful entries back for a spoken "confirm", and seals
> everything into a **signed, hash-chained, tamper-evident log**. Anyone can later prove the record wasn't
> changed — or see exactly which line was.

Hackathon prompt: *Field-to-Record: Voice-to-Structured Clinical Note Generator* (speech → NER → JSON/FHIR →
document, edge/offline, no real PHI). Our angle: **the first link in the audit chain must be trustworthy.**

**Not for clinical use.** Doses come from the NASEMSO National Model EMS Clinical Guidelines v3.0 (2022) and are
not yet line-by-line verified (`"verified_against_source": false` in `protocol.json`).

---

## 1. Status at a glance

| Phase | What | Status | Owner |
|---|---|---|---|
| 1 | Event schema, protocol file (13 drugs), eval set (119 dictations), scorer | **Done** | Nakshatra |
| 2 | Offline ASR + VAD + vocab biasing, rule + LLM hybrid extraction, spoken confirm/cancel | **Done** | Nakshatra |
| 3 | Signed hash-chained audit log, encrypted audio, verifier, tamper demo, corrections | **Done** | Nakshatra |
| 4 | **Guardian**: live protocol checks + spoken alerts + overrides into the chain | **TODO** | friend A |
| 5 | **Sync + server anchor + verifier dashboard + FHIR + handover PDF** | **TODO** | friend B |
| 6 | Hardening, eval under noise, threat model, demo video | TODO | everyone |

Tests: `python -m pytest -q` → **all green** (100+ tests). Keep it that way: every PR must pass.

---

## 2. Architecture

```
 mic ──► Silero VAD ──► clip (16 kHz wav) ──► SHA-256 ──► AES-GCM encrypt to disk
              │                                   │
              ▼                                   ▼
       faster-whisper (offline,           utterance entry ─────────────┐
       drug-vocab prompt/hotwords)                                     │
              │                                                        │
              ▼                                                        │
       LexiconCorrector ("blue cannon" → glucagon, "Roman 4" → IV)     │
              │                                                        │
              ▼                                                        ▼
   ┌──── HybridExtractor ────┐                               ┌──────────────────────┐
   │ rules  ◄─cross-check─►  │──► events (+needs_confirmation)│  AUDIT CHAIN (SQLite) │
   │ local LLM (llama.cpp,   │                               │  append-only, each    │
   │ grammar-constrained JSON)│                               │  entry: canonical JSON│
   └─────────────────────────┘                               │  → SHA-256 → Ed25519  │
              │                                               │  → prev_hash link     │
              ▼                                               └──────────────────────┘
   "Confirm fentanyl 50 micrograms I V?" ── medic: "confirm"/"haan" ── confirmation entry ─┘
              │
              ▼
   [PHASE 4] Guardian rules ── spoken alert ── override + reason ── alert / alert_response entries
              │
              ▼
   [PHASE 5] sync → server verifies + countersigns head → dashboard / FHIR / PDF
```

---

## 3. Setup (macOS Apple Silicon, tested)

```bash
cd blackbox
/opt/homebrew/bin/python3.12 -m venv .venv      # Python >= 3.10 required (code uses `X | Y` types)
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

If pip crashes with `truststore ... invalid literal for int()`, your shell environment is breaking Python's `pyexpat`
(conda/DYLD vars). Run `conda deactivate` until `(base)` disappears and
`unset DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH PYTHONHOME PYTHONPATH`, then retry. Always use `python -m pip`.

Models (download once, then everything is offline; they're git-ignored):

```bash
python -m pip install huggingface_hub
hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models
# faster-whisper downloads its model ("small", ~460 MB) automatically on first use
```

**macOS mic:** System Settings → Privacy & Security → Microphone → enable your terminal app.

---

## 4. Quickstart — things you can run right now

```bash
# Rehearse the pitch without a mic: builds a cardiac-arrest incident in the real chain
python -m audit.demo
python -m audit.cli pubkey                                   # pin device_pubkey.txt ("server side")
python -m audit.cli show --incident DEMO-ARREST              # effective record (confirmed / cancelled / corrected)
python -m audit.cli verify --pubkey device_pubkey.txt        # all green

# Play the attacker (always on a COPY of the DB)
python -m audit.cli tamper edit --seq 8 --path event.dose --value 500
python -m audit.cli verify --db runs/blackbox.tampered.db    # entry turns red: CONTENT ALTERED
python -m audit.cli tamper edit --seq 8 --path event.dose --value 500 --rehash
python -m audit.cli verify --db runs/blackbox.tampered.db    # SIGNATURE INVALID (no device key)
python -m audit.cli tamper delete --seq 5 | swap --seq 4 --with 6 | clock --seq 7 --minutes -15
python -m audit.cli tamper audio --seq 3 --from-seq 9        # follow the printed verify command
python -m audit.cli tamper truncate --keep 10                # only caught with --expect-head (server anchor)

# Honest fix = new entry, never an edit
python -m audit.cli correct --event DEMO-ARREST-0010-e1 --set dose=0.8 --reason "said point eight" --by P1

# Live, airplane mode on
python -m asr.live --mic --model small --extract --llm-model models/qwen2.5-1.5b-instruct-q4_k_m.gguf --speak
python -m asr.live --file some_recording.wav --extract      # replay a recording through the same pipeline
```

`--seq` numbers depend on what's already in `runs/blackbox.db` — read them off `verify` first, or `rm -rf runs/`
for a clean demo.

---

## 5. Repo map — what each file does

### Core data
| File | What it is |
|---|---|
| `schema.py` | Pydantic `ClinicalEvent` (medication/procedure/vital/note/**alert/alert_response/correction**), `PatientContext`, `AlertDetails`, `AlertResponseDetails` (override requires a reason). Canonical bytes. |
| `protocol.json` | 13 drugs: aliases (incl. Indian brands: Ondem, Ecosprin, Asthalin…), per-indication routes, adult dose min/max, max total, max doses, **min repeat interval**, pediatric per-kg + caps, route overrides, machine-checkable contraindications (`vital`/`flag`/`allergy`/`age`), `dangerous_mismatches` (epi 1 mg/mL IV). |
| `protocol.py` | Loads + validates protocol; `ProtocolBook.resolve_drug("epi")`, `.dose_limits(drug, indication, route, patient)`, `.is_pediatric()`, `to_canonical()` unit conversion. **Phase 4 builds on this.** |
| `normalize.py` | Spoken numbers → digits ("three twenty four" → 324, "point four" → 0.4), time phrases → absolute time ("5 minutes ago", "paanch minute pehle", "at 10:42"). |

### Speech (`asr/`)
| File | What it is |
|---|---|
| `asr/vad.py` | Streaming Silero VAD (model bundled with faster-whisper) + `Segmenter` → one clip per utterance. |
| `asr/engine.py` | `FasterWhisperEngine` (prompt/hotword biasing) and `SherpaWhisperEngine` (sherpa-onnx, Android path). |
| `asr/vocab.py` | Whisper prompt + hotwords from protocol; `LexiconCorrector` (sound-alike repair near doses, never touches correct text — tested on all gold). `KNOWN_CONFUSIONS` list to extend. |
| `asr/live.py` | **The pipeline**: mic/file → VAD → ASR → fix → extract → confirm queue → chain. `Pipeline.handle()` is where Phase 4 hooks in. |

### Extraction (`extract/` + `rule_extractor.py`)
| File | What it is |
|---|---|
| `rule_extractor.py` | Deterministic regex extractor (baseline + safety net). |
| `extract/llm.py` | Local LLM (llama.cpp or Ollama) with **JSON-schema-constrained decoding**. LLM only *reads* (drug as spoken, dose, given/planned/not_given); code resolves drug, time, and checks every number appears in the transcript. |
| `extract/hybrid.py` | LLM + rules cross-check. Disagreement / inferred dose / controlled drug / ASR doubt / LLM down → `needs_confirmation`. |
| `extract/confirm.py` | "confirm/yes/haan/theek hai" vs "cancel/no/galat/scratch that" (only short, command-only utterances), read-back text, pending queue with timeout. |

### Audit chain (`audit/`)
| File | What it is |
|---|---|
| `audit/keys.py` | Ed25519 signing key + AES audio key in `~/.blackbox` (0600). `device_id` = hash of public key. |
| `audit/chain.py` | `AuditChain.append(kind, payload, incident_id)`; SQLite with UPDATE/DELETE-blocking triggers; `correct()`; `find_event()`. |
| `audit/vault.py` | AES-256-GCM clip encryption (AAD = clip path). |
| `audit/verify.py` | `verify_chain()` (hash, signature vs **pinned** key, links, gaps, clock, audio, anchor) and `materialize()` (effective record). |
| `audit/cli.py` | `verify`, `show`, `correct`, `pubkey`, `tamper …`. |
| `audit/demo.py` | Scripted demo incident. |

### Eval (`eval/`)
| File | What it is |
|---|---|
| `eval/generate_dataset.py` | 119 dictations + gold JSON (templated EN/Hinglish, cancelled, multi-event, procedures, vitals, notes) + **30 hand-written `hard_heldout` cases — never tune on these**. |
| `eval/score.py` | Per-field P/R/F1, exact match, per-tag table, **SAFETY: wrong & unflagged meds**, WER + drug/dose recall; `--label` appends to `eval/results/history.csv`. |
| `eval/run_extractor.py` | Runs rules / llm / hybrid over gold or ASR output, prints latency. |
| `eval/run_asr.py`, `synth_tts.py`, `record.py`, `add_noise.py` | ASR eval: offline TTS voices, team recordings, siren/road noise at 10/5/0 dB. |

---

## 6. Data formats (contracts between phases — don't break them)

### Event (what extractors output, what goes into `event` entries)
```json
{"event_type": "medication", "drug": "fentanyl", "drug_as_spoken": "fent", "dose": 50.0, "unit": "mcg",
 "route": "IV", "time_said": "2026-10-25T09:12:00+05:30", "time_source": "relative",
 "needs_confirmation": true, "confirm_reasons": ["controlled drug"], "event_id": "INC-…-0004-e1"}
```
`event_type` ∈ medication | procedure (`procedure`, `params.joules`) | vital (`vital: {sbp, dbp, hr, spo2, rr, gcs, bgl_mg_dl}`) | note.
`drug` is always a **protocol id** (`adrenaline`, `normal_saline`…), never the spoken word.

### Chain entry (`audit/chain.py`)
```json
{"v":1,"seq":20,"prev_hash":"…","device_id":"BB-…","boot_id":"…","ts_wall":"…Z","ts_mono_ns":…,
 "incident_id":"DEMO-ARREST","kind":"event","payload":{…}}
```
`entry_hash = sha256(canonical_json(body))`, `sig = Ed25519(entry_hash)`. Kinds in use:
`genesis`, `utterance` (text, raw_text, corrections, audio_sha256, clip_path, extraction meta),
`event` (`event_id`, `utterance_ref`, `event`), `confirmation` (`event_id`, `event_ref`, `action`
confirmed|cancelled|unconfirmed, `said`), `correction` (`event_id`, `event_ref`, `changes`, `reason`, `by`).
**Reserved for Phase 4:** `alert`, `alert_response`. **Reserved for Phase 5:** `anchor` (server receipt).

Golden rule: **never update or delete a chain row.** Every change is a new entry that points (`*_ref`) at the old one.

---

## 7. Current numbers (for the pitch)

Extraction on the 119-case gold set (`eval/results/history.csv`):

| Extractor | Headline F1 | hard_heldout exact | Sent for confirm | **Wrong & unflagged meds** |
|---|---|---|---|---|
| Rules only | 97.3 | 53.3 % | 0 % | **12 (13.8 %)** |
| Hybrid + useless LLM (worst case) | 97.3 | 53.3 % | 100 % | **0** |
| Hybrid + Qwen2.5-1.5B (partial run, 52/119 cases) | 93.0 | – | 39.6 % | 4 (7.5 %) → fixed* |
| Hybrid + Qwen2.5-3B | not scored yet | | | median latency **4.2 s** (too slow live) |

\* Those 4 were unit mismatches (`migs` → mL) that the cross-check didn't compare; **unit disagreement is now
flagged**. **TODO: re-run the full 1.5B eval** and fill this row:
```bash
python eval/run_extractor.py --extractor hybrid --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf
python eval/score.py --pred eval/preds/ext_hybrid_qwen2.5-1.5b-instruct-q4_k_m.jsonl --label "hybrid qwen1.5b"
```

ASR (lexicon corrector, worst case Whisper-tiny on robotic TTS): drug-name recall 9.5 % → 23.8 %, end-to-end
headline F1 15.7 → 32.1. **TODO:** real numbers with `small` + team recordings + noise.

Tamper detection (all verified): edited content, re-hashed forgery, deleted entry (with count), reordered entries,
backdated timestamp, swapped voice clip, whole chain re-signed with another key, tail truncation (with anchor).

---

## 8. Known issues (found in live testing — good first tasks)

1. **Whisper hallucinations on silence/noise** — live run logged "Thank you for watching!", "Hi, I'm Nexetron".
   Fix in `asr/engine.py`/`live.py`: drop segments with `no_speech_prob > 0.6` and `avg_logprob < -1.0`,
   add a blocklist of known hallucinations, maybe raise VAD `threshold` to 0.6.
2. **Read-back echo** — `say` output was picked up by the mic ("Confirm unknown drug route unknown." became an
   utterance). Fix: pause capture while speaking (half-duplex), or drop an utterance that fuzzy-matches the last
   prompt within ~5 s. Headphones avoid it in the demo.
3. **Medication with no drug** — "Give the patient." produced a flagged event with `drug=None`. Consider dropping
   medication events with no drug and no dose instead of asking to confirm them.
4. Pipeline is single-threaded: ASR + LLM (~1–4 s) run inside `handle()`. Audio is queued, not lost, but output lags.
   A worker thread would help.
5. `speaker` is always one device; no diarization (needed for "duplicate by partner" alerts — see Phase 4).
6. Protocol doses not yet verified against the NASEMSO PDF page by page.
7. Keys are files in `~/.blackbox` — production would use Secure Enclave / Android Keystore (non-exportable).

---

## 9. PHASE 4 — Guardian (second pair of ears) — guide for friend A

**Goal:** while the medic talks, check each new medication event against `protocol.json` + what was already given,
speak a short alert, and let the medic acknowledge / override with a spoken reason. **Alert and response both go into
the chain** — that's the link between the two halves of the pitch.

### Build
1. `guardian/state.py` — `IncidentState`: `PatientContext` (weight, age, allergies, flags), latest vitals, list of
   given meds (drug, canonical dose, route, time, event_id, status). Rebuild it from the chain with `materialize()`
   (cancelled events don't count; corrections apply), so state survives restarts.
2. `guardian/rules.py` — each rule = function `(event, state, book) -> list[Alert]`. Use
   `ProtocolBook.dose_limits(drug, indication, route, patient)` and `to_canonical()`. Rules:

   | rule_id | check | severity |
   |---|---|---|
   | `DOSE_ABOVE_MAX` | single dose > `max_single` (pediatric per-kg aware) | critical |
   | `DOSE_BELOW_MIN` | dose < adult `dose_min` | warning |
   | `TOTAL_ABOVE_MAX` | cumulative (incl. this) > `max_total` | critical |
   | `TOO_MANY_DOSES` | count > `max_doses` | warning |
   | `REPEAT_TOO_SOON` | minutes since last same drug < `min_repeat_interval_min` | warning |
   | `ROUTE_NOT_ALLOWED` | route not in indication routes | warning |
   | `CONTRAINDICATED` | protocol `contraindications`: vital (e.g. SBP < 90 for nitro/fentanyl/morphine), flag, allergy, age | critical |
   | `DUPLICATE` | same drug+dose within ~2 min (likely double-logged or given twice by two crew) | warning |
   | `DANGEROUS_MISMATCH` | epi 1 mg/mL given IV/IO | critical |
   | `NEEDS_WEIGHT` | pediatric drug but no weight known | info |

   Indication: events don't carry `indication` yet. Infer it (e.g. adrenaline IV/IO → cardiac_arrest, IM →
   anaphylaxis; first indication otherwise) in one helper.
3. **Patient context from voice:** "patient weight twenty kilos", "six year old", "allergic to morphine",
   "took sildenafil yesterday". Easiest: extend the LLM schema with an optional `patient` object, or add regexes in
   a small `guardian/context.py`. Vitals already arrive as `vital` events (SBP, BGL…).
4. **Hook into `asr/live.py` → `Pipeline.handle()`** right after event entries are appended:
   ```python
   for alert in guardian.check(event, state):
       a = self.chain.append("alert", {"alert_id": ..., "event_id": e["event_id"], "event_ref": ent.entry_hash,
                                       "rule_id": alert.rule_id, "severity": alert.severity,
                                       "message": alert.message}, self.incident)
       speak(alert.spoken)          # short! "Warning: adrenaline repeat in 2 minutes, minimum 3."
   ```
   Responses: reuse `extract/confirm.py` patterns — "acknowledged", "override <reason>", "cancel dose".
   **An override must include a spoken reason** (`schema.AlertResponseDetails` already enforces it) →
   `chain.append("alert_response", {"alert_id", "alert_ref", "action", "reason", "said"}, …)`.
   Cancel-dose = also append a `confirmation` with action `cancelled` for the event.
5. Extend `audit/verify.materialize()` + `audit/cli.py show` to display alerts and responses next to events.
6. **Latency target:** alert spoken < 2 s after the medic finishes the sentence (rules are instant; the LLM is the cost).

### Evaluate
- `eval/scenes.json`: ~15 scripted scenes (lines + patient context + **expected alerts** with rule_ids), e.g.
  arrest with epi at 2 min (REPEAT_TOO_SOON), 20 kg child with 1 mg epi (DOSE_ABOVE_MAX), nitro with SBP 85
  (CONTRAINDICATED), 3rd nitro then 4th (TOO_MANY_DOSES), normal scenes with **no** alerts (false-alarm check).
- `eval/score_guardian.py`: alert precision / recall per rule; target: recall ≥ 0.95 on critical, ≤ 1 false alert per scene.
- Tests in `test_guardian.py` (one per rule + "no alert on normal dosing").

### Demo moment
Cardiac arrest: "epi one milligram IV" … 2 minutes later "epi one milligram IV" → *"Warning: adrenaline repeated
after 2 minutes, minimum 3."* → medic: *"override, rhythm changed"* → `verify` shows event, alert, override, all green.
Frame it as **decision support**, never "the AI decides".

---

## 10. PHASE 5 — Sync, anchoring, verifier dashboard, outputs — guide for friend B

**Goal:** when the ambulance gets signal, upload the chain; the server verifies it, **countersigns the head**
(the anchor), and shows a timeline where each entry is green/red; export FHIR + a handover PDF.

### Server (`server/`, FastAPI + SQLite/Postgres)
1. `POST /devices/register {device_id, public_key}` → pin key (in a real deployment done once at provisioning).
2. `POST /sync {device_id, entries:[{seq, entry_hash, prev_hash, body, sig}]}` — idempotent, entries since last
   anchored seq. Server runs `audit.verify.verify_chain()` with the **pinned** key over (stored + new) entries,
   checks the new batch continues the stored chain (same `prev_hash`), and **rejects forks**: same seq, different hash
   → store as evidence + alarm.
3. Anchor receipt: server signs `device_id | seq | entry_hash | server_time` with **its own** Ed25519 key, stores it,
   returns it. Device appends it as a chain entry `kind: "anchor"` (so the device also proves it synced). Optional:
   also get an RFC 3161 timestamp from a public TSA for the head hash.
4. Truncation / rewrite detection = `audit.verify.check_anchor(entries, (seq, hash))` against the last receipt.
   This closes the two attacks local verification can't catch (tail deletion, re-signing with a stolen key after sync).
5. Audio: clips are AES-GCM encrypted with the device's audio key. Decide: (a) upload encrypted clips + wrap the
   audio key to the server's X25519 public key, or (b) keep audio on device and only fetch on dispute. (a) is better
   for the "click to play the original voice" demo.

### Device side (`sync/client.py`)
- Offline queue = just "entries with seq > last anchored seq". Retry with backoff; works after hours offline.
- CLI: `python -m sync.client push --server http://...`.

### Verifier dashboard
- Simple FastAPI + Jinja/HTMX or Next.js: incident list → timeline (time, kind, summary, ✅/❌ per entry, reasons
  from `verify_chain`), effective record (`materialize`), alerts/overrides (Phase 4), **play clip** button
  (decrypt server-side, re-hash, show "audio matches chain ✅").
- Demo: tamper the **server copy** (`audit.cli tamper …` on it) → refresh → that line turns red.

### Outputs
- **FHIR R4 Bundle** (`fhir.resources`): `MedicationAdministration` (medication = drug, dosage dose/route, effective =
  `time_said`, status from confirmation: completed / entered-in-error for cancelled / not-done), `Procedure`,
  `Observation` (vitals, LOINC codes), and **`Provenance`** per resource with `signature` = entry sig, `target` =
  resource, `entity` = chain entry hash. Corrections → updated resource + Provenance pointing at both entries.
  Map drug ids → RxNorm (small dict in protocol.json, e.g. adrenaline 3992, fentanyl 4337 — verify codes).
- **Handover PDF** (WeasyPrint or ReportLab): MIST / SBAR layout, timeline table, alerts + overrides, footer with
  head hash + anchor receipt + "verify at …" QR code.

### Tests
- Server rejects a batch with a bad signature / broken link / fork; accepts an idempotent re-push.
- Truncation after anchoring is reported. FHIR bundle validates.

---

## 11. Conventions

- Python ≥ 3.10, run everything from repo root (`python -m package.module`).
- New features come with tests; `python -m pytest -q` must stay green.
- Never tune prompts/rules by looking at `hard_heldout` cases. Add new hard cases blind (someone else writes them).
- Never commit models, `runs/`, keys, audio, or n2c2/MIMIC data (see `.gitignore`; the n2c2 DUA forbids posting data).
- Every number in the pitch should come from `eval/results/history.csv` (use `--label`).

## 12. Sources
- NASEMSO National Model EMS Clinical Guidelines v3.0 (2022): https://nasemso.org/wp-content/uploads/National-Model-EMS-Clinical-Guidelines_2022.pdf
- n2c2 datasets (DUA): https://n2c2.dbmi.hms.harvard.edu/data-sets
- PhysioNet guidance on MIMIC + LLMs (local models recommended): https://physionet.org/news/post/llm-responsible-use/