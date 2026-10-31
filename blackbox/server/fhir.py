
from __future__ import annotations

import json
from typing import Optional

RXNORM = {"adrenaline": "3992", "amiodarone": "703", "aspirin": "1191", "nitroglycerin": "4917",
          "naloxone": "7242", "midazolam": "6960", "fentanyl": "4337", "morphine": "7052",
          "ondansetron": "26225", "salbutamol": "435", "dextrose": "4850", "glucagon": "4832",
          "normal_saline": "9863"}
ROUTE_SNOMED = {"IV": ("47625008", "Intravenous"), "IO": ("60213007", "Intraosseous"),
                "IM": ("78421000", "Intramuscular"), "IN": ("46713006", "Nasal"), "PO": ("26643006", "Oral"),
                "SL": ("37839007", "Sublingual"), "NEB": ("447694001", "Respiratory tract"),
                "SC": ("34206005", "Subcutaneous"), "ODT": ("26643006", "Oral"), "ET": ("404820008", "Endotracheal")}
UCUM = {"mg": "mg", "mcg": "ug", "g": "g", "mL": "mL"}
LOINC = {"sbp": ("8480-6", "Systolic blood pressure", "mm[Hg]"), "dbp": ("8462-4", "Diastolic blood pressure", "mm[Hg]"),
         "hr": ("8867-4", "Heart rate", "/min"), "spo2": ("59408-5", "Oxygen saturation", "%"),
         "rr": ("9279-1", "Respiratory rate", "/min"), "gcs": ("9269-2", "Glasgow coma score total", "{score}"),
         "bgl_mg_dl": ("2339-0", "Glucose", "mg/dL")}
STATUS = {"recorded": "completed", "confirmed": "completed", "cancelled": "not-done",
          "pending": "unknown", "unconfirmed": "unknown"}


def _provenance(rid: str, rtype: str, entry: dict, device_id: str) -> dict:
    return {
        "resourceType": "Provenance", "id": f"prov-{rid}",
        "target": [{"reference": f"{rtype}/{rid}"}],
        "recorded": entry["ts_wall"],
        "agent": [{"type": {"text": "author device"}, "who": {"reference": f"Device/{device_id}"}}],
        "entity": [{"role": "source", "what": {"identifier": {"system": "urn:blackbox:chain-entry",
                                                               "value": entry["entry_hash"]}}}],
        "signature": [{"type": [{"system": "urn:iso-astm:E1762-95:2013", "code": "1.2.840.10065.1.12.1.1",
                                 "display": "Author's Signature"}],
                       "when": entry["ts_wall"], "who": {"reference": f"Device/{device_id}"},
                       "sigFormat": "application/octet-stream", "data": entry["sig"]}],
    }


def build_bundle(device_id: str, incident_id: str, record: list[dict], patient: dict,
                 entry_index: dict[int, dict], anchor: Optional[dict]) -> dict:
    """record = audit.verify.materialize(); entry_index = {seq: {entry_hash, sig, ts_wall}}."""
    pid = f"pt-{incident_id}"
    res: list[dict] = [{
        "resourceType": "Patient", "id": pid,
        "identifier": [{"system": "urn:blackbox:incident", "value": incident_id}],
        "extension": [{"url": "urn:blackbox:allergies", "valueString": ", ".join(patient.get("allergies") or [])}]
        if patient.get("allergies") else [],
    }, {"resourceType": "Device", "id": device_id, "identifier": [{"system": "urn:blackbox:device", "value": device_id}],
        "deviceName": [{"name": "Black Box ambulance recorder", "type": "user-friendly-name"}]}]
    if patient.get("weight_kg"):
        res.append({"resourceType": "Observation", "id": f"weight-{incident_id}", "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "29463-7", "display": "Body weight"}]},
                    "subject": {"reference": f"Patient/{pid}"},
                    "valueQuantity": {"value": patient["weight_kg"], "unit": "kg", "system": "http://unitsofmeasure.org",
                                      "code": "kg"}})
    for x in record:
        ev, entry = x["event"], entry_index.get(x["seq"])
        rid = x["event_id"].replace("_", "-")
        notes = [f"Guardian {a['rule_id']}: {a['message']} -> "
                 f"{(a['response'] or {}).get('action', 'no answer')}"
                 + (f" ({a['response']['reason']})" if a.get("response") and a["response"].get("reason") else "")
                 for a in x.get("alerts", [])]
        notes += [f"corrected {h['before']} -> {h['after']}: {h.get('reason')} (by {h.get('by')})"
                  for h in x["history"] if h["what"] == "corrected"]
        if ev["event_type"] == "medication":
            r = {"resourceType": "MedicationAdministration", "id": rid, "status": STATUS.get(x["status"], "unknown"),
                 "medicationCodeableConcept": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                                                           "code": RXNORM.get(ev.get("drug"), ""),
                                                           "display": (ev.get("drug") or "").replace("_", " ")}],
                                               "text": ev.get("drug_as_spoken") or ev.get("drug")},
                 "subject": {"reference": f"Patient/{pid}"},
                 "effectiveDateTime": ev.get("time_said"),
                 "dosage": {}}
            if ev.get("route") in ROUTE_SNOMED:
                c, disp = ROUTE_SNOMED[ev["route"]]
                r["dosage"]["route"] = {"coding": [{"system": "http://snomed.info/sct", "code": c, "display": disp}],
                                        "text": ev["route"]}
            if ev.get("dose") is not None:
                u = ev.get("unit") or ""
                r["dosage"]["dose"] = {"value": ev["dose"], "unit": u, "system": "http://unitsofmeasure.org",
                                       "code": UCUM.get(u, u)}
            rtype = "MedicationAdministration"
        elif ev["event_type"] == "procedure":
            r = {"resourceType": "Procedure", "id": rid, "status": "completed" if x["status"] != "cancelled" else "not-done",
                 "code": {"text": (ev.get("procedure") or "").replace("_", " ")},
                 "subject": {"reference": f"Patient/{pid}"}, "performedDateTime": ev.get("time_said")}
            if ev.get("params", {}).get("joules"):
                notes.append(f"energy {ev['params']['joules']} J")
            rtype = "Procedure"
        elif ev["event_type"] == "vital":
            r = {"resourceType": "Observation", "id": rid, "status": "final",
                 "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                           "code": "vital-signs"}]}],
                 "code": {"text": "Vital signs"}, "subject": {"reference": f"Patient/{pid}"},
                 "effectiveDateTime": ev.get("time_said"),
                 "component": [{"code": {"coding": [{"system": "http://loinc.org", "code": LOINC[k][0],
                                                     "display": LOINC[k][1]}]},
                                "valueQuantity": {"value": v, "unit": LOINC[k][2], "system": "http://unitsofmeasure.org",
                                                  "code": LOINC[k][2]}}
                               for k, v in ev.get("vital", {}).items() if k in LOINC]}
            rtype = "Observation"
        else:
            continue
        if notes:
            r["note"] = [{"text": n} for n in notes]
        res.append(r)
        if entry:
            res.append(_provenance(rid, rtype, entry, device_id))
    bundle = {"resourceType": "Bundle", "type": "collection", "timestamp": anchor["server_time"] if anchor else None,
              "identifier": {"system": "urn:blackbox:incident", "value": incident_id},
              "entry": [{"fullUrl": f"urn:uuid:{r['resourceType']}-{r['id']}", "resource": r} for r in res]}
    if anchor:
        bundle["meta"] = {"tag": [{"system": "urn:blackbox:anchor", "code": f"{anchor['seq']}:{anchor['entry_hash']}"}]}
    return json.loads(json.dumps(bundle))