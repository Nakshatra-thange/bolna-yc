# Black Box: test everything (English + Hindi/Hinglish)

Run from the repo root with the venv active. Order matters: do 1 → 5.

## 1. Automated checks (2 minutes)
```bash
python -m pytest -q                                   # expect: 172 passed
python eval/score_guardian.py                         # expect: 23/23 scenes, precision 1.0, recall 1.0
python eval/run_extractor.py --extractor rules && python eval/score.py --pred eval/preds/ext_rules.jsonl --label rules
```
Pass = all green, and the SAFETY line shows 0 wrong-and-unflagged medications.
(If a file name differs, `ls eval/preds/` and use the one just written.)

## 2. English, typed (no mic needed, deterministic)
```bash
python - <<'PY'
from datetime import datetime, timezone
from rule_extractor import extract
now = datetime.now(timezone.utc)
for s in ["gave adrenaline one milligram IV",
          "aspirin three hundred milligram oral",
          "BP one twenty over eighty heart rate ninety",
          "defibrillate at two hundred joules",
          "gave morphine ten milligram IV ten minutes ago"]:
    print(s, "->", [{k: v for k, v in e.items() if k in ("event_type","drug","dose","unit","route","vital","procedure")} for e in extract(s, now)])
PY
```
Pass = correct drug, dose, unit, route, vitals in every line.

## 3. Hindi / Hinglish, typed (romanized Hindi)
```bash
python - <<'PY'
from datetime import datetime, timezone
from rule_extractor import extract
from guardian.context import parse_context
now = datetime.now(timezone.utc)
for s in ["adrenaline ek milligram IV diya",
          "morphine do milligram diya patient ko",
          "fentanyl pachaas mcg diya",
          "BP ek sau bees by assi",
          "aspirin teen sau milligram khilaya"]:
    print(s, "->", [{k: v for k, v in e.items() if k in ("event_type","drug","dose","unit","route","vital")} for e in extract(s, now)])
for s in ["wazan pachaas kilo, umar das saal", "patient ko penicillin ki allergy hai"]:
    print(s, "->", parse_context(s))
PY
```
Pass = doses 1 / 2 / 50 / 300, BP 120/80, weight 50, age 10, allergy penicillin.
Known limit: ambiguous words ("do", "ek") alone are NOT turned into a dose (safe on purpose);
they need a unit next to them ("do milligram").

## 4. Live voice demo (English)
```bash
python -m asr.live --mic --model small --extract --llm-model models/qwen2.5-1.5b-instruct-q4_k_m.gguf --speak
```
Say, with a short pause between sentences:
1. "Patient weight seventy kilos, allergic to penicillin"
2. "Gave adrenaline one milligram IV"            → device reads it back → say "confirm"
3. "Gave morphine ten milligram IV"              → watch for a Guardian alert → say "override, reason: severe pain" (or "cancel")
4. "Gave adrenaline one milligram IV" again within 45 s → duplicate alert.
Pass = confirmations appear in `python -m audit.cli show --incident <id>`, alerts have answers.

Hindi live: Whisper with `--language hi` writes Devanagari, which our text layers do not read yet.
For tomorrow's live demo keep `--language en` and speak Hinglish in Roman style; for Hindi *accuracy* the plan is Sarvam ASR (see README roadmap).

## 5. Tamper-evidence + server demo
```bash
pip install fastapi uvicorn httpx
rm -rf runs server_data && python -m audit.demo
python -m server.app                       # terminal 1, open http://127.0.0.1:8000
python -m sync.client push                 # terminal 2
```
In the page: Attack simulation → "edit dose" → Run attack (banner turns red, names the entry) → Restore (green again).
Then: `python -m audit.cli tamper edit --seq 8 --path event.dose --value 500` and `python -m audit.cli verify --db runs/blackbox.tampered.db`.

## Common errors
| Symptom | Cause / fix |
|---|---|
| `git push` 2.9 GiB | models/venv committed. `git reset --soft origin/main; git rm -r --cached .venv models runs server_data; commit; push` |
| pytest "import file mismatch" | stale `eval/test_guardian.py` or `__pycache__`: delete them |
| live ASR ~6 s per 1 s audio | x86 Python under Rosetta. `python -c "import platform;print(platform.machine())"` must print `arm64`; rebuild venv with `/opt/homebrew/bin/python3.12`; use `--model base` |
| device transcribes its own read-back | update `asr/live.py` (echo gate) |
| "server key changed" on sync | you wiped `server_data` but kept `runs/server_pubkey.txt`: `rm runs/server_pubkey.txt` |
| "device key pinned" 409 on register | you regenerated `runs/` keys but kept `server_data`: wipe both together |
| `ModuleNotFoundError: fastapi` | `pip install fastapi uvicorn httpx` |