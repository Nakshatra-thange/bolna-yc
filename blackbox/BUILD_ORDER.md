Build plan in 6 phases
Phase 1: Foundations and evaluation set

Lock the scope before writing code.

Event schema. Define the fields: event_type (medication/procedure/vital/note), drug, dose, unit, route, time said, time logged, speaker, confidence, and the audio clip’s hash.
Protocol file. Write a JSON file covering 12–15 common EMS drugs, for example:
adrenaline, amiodarone, aspirin, nitroglycerin, naloxone, midazolam
fentanyl or morphine, ondansetron, salbutamol, dextrose, normal saline
For each drug: dose ranges, per-kg limits, maximum total dose, minimum repeat interval and key contraindications. Take these from a public EMS protocol and cite it.
Evaluation set. Write 60–100 synthetic dictations with ground-truth JSON. Record some of them yourselves, both clean and with siren or road noise, and include a few in Hinglish.
Data access. Apply for n2c2 access now, because approval takes days.
Scoring script. Measure speech-recognition error (WER) and per-field accuracy (F1) for drug, dose, route and time.

Done when: an evaluation script runs and prints a score, even if it’s 0.

Phase 2: Speech and extraction engine (offline)
Speech-to-text runs locally, with voice-activity detection to cut speech into clips.
Bias the speech recognizer toward your drug vocabulary using its prompt or hotwords.
Add a normalizer that turns spoken forms into clean values: “zero point five milligram” becomes 0.5 mg, “one in ten thousand” becomes a concentration, and “five minutes ago” becomes an absolute time.
Use a small local language model with grammar-constrained JSON output, so it can’t produce malformed records. Add a rule-based fallback for very common phrases.
Mark low-confidence extractions as “needs confirmation.” The medic can then say “confirm” or “cancel.”

Done when: speech goes in and correct JSON comes out on the laptop with Wi-Fi off, and you can quote field accuracy and latency.

Phase 3: Audit chain core (the heart of the pitch)
Entry format. Each entry is canonical JSON (RFC 8785) containing {seq, prev_hash, device_id, timestamp, monotonic_counter, payload, audio_sha256}.
Hashing and signing. Hash each entry with SHA-256 and sign it with Ed25519. Keep the key in the platform’s protected storage (a TPM or OS keyring on a laptop, or Android Keystore).
Three linked entries per utterance:
the raw transcript
the extracted event
the confirmation
This way, even a wrong extraction can be traced back to what was actually said.
Corrections. A correction is a new entry that points to the one it amends. Nothing is ever deleted.
Audio. Store the clips encrypted on the device. Only their hashes go into the chain.

Done when: you can change one byte in the database and the verifier pinpoints the broken entry.

Phase 4: Guardian (the second pair of ears)
Live patient state: weight, age, allergies, and every drug given so far with its time.
Rules engine reading from the protocol file, with these checks:
single dose too high
total dose over the maximum
repeat dose given too soon
duplicate across crew members
allergy or contraindication conflict
per-kg dose for children
Alerts: a short spoken alert plus an on-screen card.
Medic response. The medic can acknowledge or override with a spoken reason. The alert and the override are both chained entries. This is the link between the two halves of the product.

Done when: dictating an overdose triggers an alert within about 2 seconds, and the alert appears in the chain.

Phase 5: Sync, anchoring, verifier and outputs
Offline queue. Entries wait on the device and sync when a connection returns.
Server verification. The server checks the whole chain, then countersigns the latest entry (the “chain head”) with a server timestamp. Optionally, add a public timestamp authority (RFC 3161). This addresses two attacks:
Clock tampering: the device clock can lie, but the server timestamp is independent.
Tail deletion: the server already holds the last head it saw, so deleted entries at the end are detectable.
Verifier dashboard: a timeline of the incident with a green or red integrity status on each entry. Clicking an entry plays the original voice clip, alongside its extracted fields.
Outputs:
a FHIR Bundle with MedicationAdministration and Procedure resources, plus Provenance resources (FHIR’s built-in way to record who did what, and to carry signatures)
a handover PDF for the hospital

Done when: you can tamper with a record on the server and the dashboard shows exactly what changed.

Phase 6: Hardening, evaluation and demo
Run the evaluation set under noise. Report speech error, field accuracy, alert precision and recall, and end-to-end latency.
Write a threat-model table covering four attacks: editing, deleting, reordering and clock tampering. Show how each one is detected.
Run a “what if the device is stolen” check: is the key protected, is the audio encrypted?
Demo script, about 3 minutes:
Switch on airplane mode.
Run a chaotic cardiac-arrest scene.
The Guardian catches an early repeat adrenaline dose, and the medic overrides it with a spoken reason.
Reconnect and sync.
A “bad actor” edits the fentanyl dose on the server, and the verifier turns that line red.
Play the original voice clip as evidence.
Keep the whole last day for the demo video.



3. The stack

 Your requirement is that everything runs offline on one device, so there’s nothing to transport. Mention it only as a later extension, for live consultation with a remote doctor once the ambulance has signal.

You need speech-to-text (voice to text) and text-to-speech (for spoken alerts), both running locally:

| Layer | Pick | Why |
|---|---|---|
| Voice activity detection | Silero VAD | Small and accurate; cuts speech into clips |
| Speech-to-text | faster-whisper (small or base, int8) or whisper.cpp | Strong accuracy offline; can be biased with a drug-name prompt |
| Lower-latency alternative | Moonshine, or sherpa-onnx streaming models | Faster on weak hardware |
| Fixed command words | Vosk with a restricted vocabulary | Reliable “confirm”, “cancel”, “override” |
| Text-to-speech for alerts | Piper, or Kokoro | Fast, natural-sounding, fully offline |
| All-in-one option | sherpa-onnx | Speech-to-text, VAD and text-to-speech in one toolkit, with Android bindings |
| Extraction | Small instruct model (roughly 1–4B parameters, Qwen or Phi family) in GGUF format via llama.cpp, with grammar or JSON-schema constrained output | Always-valid JSON; runs on a CPU |
| Fallback extraction | spaCy or medspaCy rules plus your drug lexicon | Deterministic safety net |
| Guardian | Python rules engine reading the protocol JSON | Auditable; doctors can read the rules |
| Crypto | PyNaCl (Ed25519), hashlib (SHA-256), a canonical-JSON library, AES-GCM for audio | Standard primitives |
| Storage on device | SQLite, append-only (triggers block updates and deletes) | Simple and robust offline |
| Backend and verifier | FastAPI + Postgres; React or Next.js dashboard | Quick to build |
| FHIR / PDF | fhir.resources / WeasyPrint | Standards-compliant output |
| Evaluation | jiwer for speech error, plus a custom per-field F1 script | Numbers for the pitch |

Deployment recommendation: run the edge unit on a laptop or mini-PC that stands in for the ambulance’s onboard computer. It runs everything locally and hosts a web UI. A phone or tablet connects to that UI over the laptop’s own Wi-Fi hotspot, with no internet at all. This is realistic, since ambulances already carry onboard tablets, and far easier than fitting a language model onto a phone. If you have spare time, port the speech-to-text and text-to-speech to Android with sherpa-onnx.

Want me to turn this into a shareable doc for your team, or draft the event schema and protocol JSON next?