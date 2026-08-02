# Case-input JSON schema

This document describes the JSON schema used as the model input in the
MECR-RAG pipeline. The schema is populated by an in-house Python script
that maps pre-existing structured EHR triage fields directly into the
schema. No clinician-guided cleaning, correction, or content enrichment is
performed at the schema-mapping stage.

## Top-level structure

```json
{
  "Demographics": { ... },
  "Clinical Presentation": { ... },
  "Vitals and Observations": { ... }
}
```

Both space-delimited and underscore-delimited key variants are accepted at
load time (for example, `Clinical Presentation` or `Clinical_Presentation`)
so the schema can consume the field names used by different EHR export
paths.

## Fields

### Demographics
| Field | Type | Missing encoding | Notes |
|-------|------|------------------|-------|
| Age | integer or `""` | `""` | Years |
| Sex | `"M"` / `"F"` / `""` | `""` | |
| Ambulatory status | string | `""` | Free text (e.g., "walking", "wheelchair") |
| Risk of fall | string | `""` | Free text |
| Informant | string | `""` | Who provided the history |
| Communication | string | `""` | Language / communication notes |
| Allergies | string | `""` | Free text |
| ADR | string | `""` | Adverse drug reaction history |
| Alerts | string | `""` | EHR alert flags |
| Past health | string | `""` | Past medical history free text |
| TOCC | string | `""` | Travel / Occupation / Contact / Cluster (Hong Kong epidemiological screen) |
| Referral (if any) | string | `""` | Source of referral if applicable |

### Clinical Presentation
| Field | Type | Missing encoding | Notes |
|-------|------|------------------|-------|
| Chief complaint | string | `""` | Free text |
| Condition on Arrival | string | `""` | Free text |

### Vitals and Observations

`Vital Signs` sub-object:
| Field | Type | Missing encoding | Notes |
|-------|------|------------------|-------|
| GCS | string | `""` | e.g., "15/15" |
| BP | string | `""` | Systolic/diastolic, mmHg (e.g., "148/82") |
| PR/AR | string | `""` | Pulse rate / apex rate, bpm |
| Temp | string | `""` | Temperature, °C |
| SpO2 | string | `""` | Oxygen saturation, % (with support noted if not room air) |
| RR | string | `""` | Respiratory rate, breaths/min |
| PFR | string | `""` | Peak flow rate |
| Limbs | string | `""` | Limb power / neurological findings |
| Pupil | string | `""` | Pupil size / reactivity |

`Triage Intervention` sub-object:
| Field | Type | Missing encoding | Notes |
|-------|------|------------------|-------|
| H'stix | string | `""` | Haemoglucostix / capillary blood glucose |
| H'cue | string | `""` | Haemocue / haemoglobin at triage |

`Menstruation / Pregnancy Status`: string | `""`.

## Missing values

Missing measurements are represented as empty strings (`""`). No
statistical imputation is performed at any stage of the pipeline.

## Preprocessing pipeline

Raw EHR triage entry → in-house Python schema-mapping script (this JSON
structure) → LLM-generated concise case summary (pipeline step, not
retrospective pre-cleaning) → triage prompt.

The LLM summary step is *part of the pipeline*, not a manual pre-cleaning
stage. During summarisation the LLM is instructed to interpret medical
abbreviations contextually, correct obvious typos based on medical
context, and expand a small set of Hong Kong facility abbreviations (e.g.,
`PMH` → `Princess Margaret Hospital`). This step runs identically on
prospective inputs and does not confer any retrospective advantage.

## Redacted worked example

Illustrative — not a real patient record.

**Raw JSON input:**
```json
{
  "Demographics": {
    "Age": 76,
    "Sex": "F",
    "Ambulatory status": "walking",
    "Past health": "HT, DM, CKD stage 3",
    "TOCC": ""
  },
  "Clinical Presentation": {
    "Chief complaint": "SOB x 2 days, worse on exertion",
    "Condition on Arrival": "Tachypnoeic, using accessory muscles"
  },
  "Vitals and Observations": {
    "Vital Signs": {
      "GCS": "15/15",
      "BP": "148/82",
      "PR/AR": "118",
      "Temp": "37.1",
      "SpO2": "88 (RA)",
      "RR": "28",
      "Limbs": "",
      "Pupil": ""
    },
    "Triage Intervention": {
      "H'stix": "",
      "H'cue": ""
    }
  }
}
```

**Pipeline LLM summary (illustrative):**
> "76 y F with hypertension, diabetes, and chronic kidney disease
> presenting with 2-day worsening shortness of breath. On arrival
> tachypnoeic and hypoxic (SpO2 88% on room air, RR 28, HR 118); alert
> (GCS 15). No fever."

## Prospective applicability

The same script and the same LLM summariser run identically on live EHR
triage fields without modification. The analytic cohort therefore did not
receive any retrospective input-quality advantage over prospective
attendances beyond what the same automated pipeline would produce in real
time.
