1. Alerts, overrides and corrections are event types. They go into the same chain as medications, which is what links the Guardian to the audit trail. An override can’t be saved without a spoken reason.

2. Every medication or procedure must carry its audio hash. This is the chain of custody from voice to record.

3. There are two timestamps. time_said is when the action happened (it can be resolved from “five minutes ago”). time_logged is when the device captured it. The schema rejects a time_said later than time_logged, and rejects timestamps without a timezone.

4. Contraindications can be checked by code. They are rules like sbp < 90 or the flag pde5_inhibitor_48h, not just text, so the Guardian can run them directly in Phase 4.

5. The protocol is validated when it loads. Dose ranges in the wrong order, a missing source, or the same alias on two drugs will refuse to load.

6. Indian brand names are included as aliases, for example Ecosprin, Ondem, Asthalin and Sorbitrate. This helps with the code-mixed speech idea later.

7. Sources -- NASEMSO – Model Clinical Guidelines Version 3
NASEMSO V3 PDF (Iowa HHS mirror)
NASEMSO Version 4 stakeholder feedback survey
Wyoming OEMS – Scope of Practice / Model Guidelines