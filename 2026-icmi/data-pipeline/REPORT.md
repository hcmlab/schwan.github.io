# ICEP Annotation Data Preparation — Technical Report

**Project:** Qwen-Omni Finetuning for Infant-Caregiver Engagement Phase (ICEP) Classification  
**Date:** 2026-03-05  
**Data Source:** `X:\data\Schwan_T3_Clean` (87 sessions)  
**Output:** `X:\data\Schwan_T3_FineTune`

---

## 1. Objective

Prepare human-annotated ICEP engagement data from the Schwan T3 dataset for finetuning **Qwen-Omni**, a multimodal model capable of processing audio-video input. The pipeline converts proprietary `.xiact` annotation files into structured JSON suitable for LlamaFactory-based training.

## 2. ICEP Annotation Standard

The **Infant Coding of Engagement Phases** (ICEP) system codes mutually exclusive engagement phases for both infant and caregiver using facial expressions, gaze direction, body posture, and vocalizations.

### Infant Codes (6 codes)

| Code | Label | Description |
|------|-------|-------------|
| `Inon` | Object/Environment Engagement | Infant focused on objects or environment, neutral/positive affect |
| `Ineu` | Social Monitor | Infant looking at caregiver with neutral expression |
| `Ipos` | Social Positive Engagement | Infant engaged with caregiver showing positive affect (smiles, coos) |
| `Ipro` | Protest | Negative engagement with active protest (crying, pushing away) |
| `Iwit` | Withdrawn | Negative engagement with passive withdrawal |
| `Iusc` | Unscorable | Segment cannot be reliably coded |

### Caregiver Codes (6 codes)

| Code | Label | Description |
|------|-------|-------------|
| `Cpvc` | Social Monitor / Positive Vocs | Caregiver monitoring with positive vocalizations |
| `Cneu` | Social Monitor / No Vocs | Caregiver monitoring without affective vocalizations |
| `Cpos` | Social Positive Engagement | Caregiver positive, warm engagement |
| `Cint` | Intrusive | Caregiver overriding infant's focus/activity |
| `Cnon` | Non-Infant Focused | Caregiver not focused on infant |
| `Cusc` | Unscorable | Segment cannot be reliably coded |

### Still Face Paradigm (SFP) Phases

Each session follows the Still Face Paradigm with expected durations:

- **Play** ≈ 2 min — Normal interaction
- **Transition** — Variable duration
- **Still Face (SFP)** ≈ 2 min — Caregiver maintains neutral, unresponsive face
- **Reunion (RP)** ≈ 2 min — Normal interaction resumes

---

## 3. Data Pipeline

### 3.1 Scripts

| Script | Purpose |
|--------|---------|
| `00_prepare_annotations.py` | Parses `.json` annotations, enriches with ICEP descriptions, generates gap segments, creates per-session finetune annotation JSONs |
| `validate_xiact_vs_json.py` | Validates `.xiact` → `.json` conversion integrity across all 87 sessions |
| `fix_json_from_xiact.py` | Repairs conversion errors by re-parsing `.xiact` with correct column mapping |
| `annotation_stats.py` | Generates annotation distribution statistics and visualizations |
| `session_stats.py` | Generates per-session coverage analysis, SFP phase timeline, and anomaly detection |
| `01_extract_chunks.py` | Extracts audio-video chunks aligned with annotations |
| `02_create_dataset.py` | Creates LlamaFactory-compatible dataset entries |
| `03_split_data.py` | Splits data into train/val/test sets |
| `04_verify_omni_format.py` | Validates final dataset format for Qwen-Omni |

### 3.2 Enriched Annotation Format

Each annotation in the finetune JSONs contains:

```json
{
  "code": "Inon",
  "label": "Object/Environment Engagement",
  "track": "Infant_Engagement",
  "full_description": "The infant is focused on objects...",
  "video_description": "Look for the infant's gaze...",
  "audio_description": "Listen for neutral or quiet...",
  "start": 72.6,
  "end": 101.6,
  "buffered_start": 69.6,
  "buffered_end": 104.6
}
```

- **Buffer:** ±3.0 seconds around each annotation for model context
- **Gap segments:** Unannotated regions (>1s) are included as `no_annotation` to teach the model areas without specific codes

---

## 4. Data Validation & Fix

### 4.1 Conversion Issue Discovered

The `.xiact` → `.json` conversion contained errors in **8 out of 87 sessions**. The root cause was a **swapped column order** in the `.xiact` header:

- **Expected:** `Caregiver_Engagement_Phases | Infant_Engagement_Phases | Add_Infant_Codes`
- **Actual (8 files):** `Caregiver_Engagement_Phases | Add_Infant_Codes | Infant_Engagement_Phases`

The original converter used fixed column positions rather than parsing header names, causing it to put additional infant codes (`Isc o`, `Isc h`, `Idis`) into the `Infant_Engagement` track while **dropping all primary infant engagement annotations**.

### 4.2 Affected Sessions

| Session | Missing Infant Events | Impact |
|---------|----------------------:|--------|
| CAOKKA01_MUC_T3 | 165 | Coverage dropped from 95% → 6% |
| ELFEAN01_HD_T3 | 99 | Most infant codes lost |
| BEMASU01_HD_T3 | 74 | Most infant codes lost |
| BESEHE01_HD_T3 | 83 | Most infant codes lost |
| JEAPBA01_HD_T3 | 84 | Coverage dropped to ~2% |
| CHMAMA01_HD_T3 | 78 | Most infant codes lost |
| JEAPCO01_HD_T3 | 50 | Coverage dropped to ~15% |
| JOAUUN01_HD_T3 | 46 | Coverage dropped to ~0% |

### 4.3 Fix Applied

`fix_json_from_xiact.py` re-parses each `.xiact` using the **actual header column names** (not assumed positions), extracts the correct tracks, and patches the `.json` files. Original files are backed up to `X:\data\Schwan_T3_Clean\_backups\`.

### 4.4 Post-Fix Validation

After repair, `validate_xiact_vs_json.py` confirmed:

> ✅ **ALL 87 SESSIONS MATCH PERFECTLY — 0 discrepancies**

Caregiver tracks were unaffected (all 87 matched from the start). The fix restored all infant annotations and brought coverage back to expected levels.

---

## 5. Dataset Statistics

### 5.1 Overall Numbers

| Metric | Value |
|--------|-------|
| Total sessions | 87 |
| Total annotations | 11,563 |
| Avg annotations/session | 132.9 |
| Total video duration | 11.5 hours |
| Mean session duration | 7.9 min |
| Mean annotation coverage | 85% |
| Sessions <50% coverage | 6 |

### 5.2 Annotation Distribution

| Code | Label | Count | % | Avg Duration |
|------|-------|------:|---:|---------:|
| Cpvc | Social Monitor / Positive Vocs | 2,872 | 24.8% | 3.6s |
| Inon | Object/Environment Engagement | 1,996 | 17.3% | 10.3s |
| Cneu | Social Monitor / No Vocs | 1,816 | 15.7% | 7.8s |
| Cpos | Social Positive Engagement | 1,705 | 14.7% | 2.3s |
| Ineu | Social Monitor | 1,669 | 14.4% | 2.5s |
| Ipro | Protest | 475 | 4.1% | 10.3s |
| Ipos | Social Positive Engagement | 460 | 4.0% | 2.5s |
| no_annotation | No Annotation | 364 | 3.1% | 61.8s |
| Iusc | Unscorable | 60 | 0.5% | 2.3s |
| Cusc | Unscorable | 57 | 0.5% | 4.1s |
| Cint | Intrusive | 47 | 0.4% | 3.3s |
| Cnon | Non-Infant Focused | 41 | 0.4% | 2.9s |
| Iwit | Withdrawn | 1 | 0.0% | 6.9s |

**Track split:** Caregiver 58.6% (6,781) vs Infant 41.4% (4,782)

### 5.3 Annotation Distribution Visualization

![ICEP Annotation Distribution — 87 sessions, 11,563 annotations](X:\data\Schwan_T3_FineTune\stats\annotation_distribution.png)

### 5.4 Notable Observations

- **Class imbalance:** `Iwit` (Withdrawn) has only 1 instance across all sessions — this code is extremely rare and may need data augmentation or special handling
- **`Ipro` (Protest)** and **`Ipos` (Social Positive)** are underrepresented (4% each) compared to dominant codes `Cpvc`/`Inon`/`Cneu` which account for 58% of all annotations
- **`no_annotation` segments** average 61.8s duration — these are large unannotated gaps, mostly from long reunion phases in sessions with extended video recordings

---

## 6. Session-Level Analysis

### 6.1 Coverage Analysis

![Session-by-Session Analysis](X:\data\Schwan_T3_FineTune\stats\session_analysis.png)

### 6.2 Sessions with Low Coverage (<50%)

These sessions have genuinely low annotation coverage (not a conversion error):

| Session | Duration | Coverage | Root Cause |
|---------|----------|----------|------------|
| SAJARE01_HD_T3 | 38.2 min | 15% | Reunion phase = 2044s (video not trimmed) |
| MAMABE01_HD_T3 | 29.7 min | 21% | Reunion phase = 1535s (video not trimmed) |
| ANOKPE01_HD_T3 | 22.0 min | 26% | Reunion phase = 1085s (video not trimmed) |
| ANSEBE01_HD_T3 | 17.3 min | 28% | Reunion phase = 687s + short SFP (36s) |
| VEJAGE01_MUC_T3 | 20.4 min | 32% | Reunion phase = 964s (video not trimmed) |
| NAAPCA01_HD_T3 | 12.3 min | 50% | Reunion phase = 491s (video not trimmed) |

All low-coverage sessions share the same pattern: the video recording continued long after the reunion phase ended, inflating total duration while annotations correctly cover only the experimental portion.

### 6.3 SFP Phase Anomalies

- **MUC sessions** tend to have longer play phases (~3 min vs expected ~2 min)
- **HD sessions** are more consistent with the expected ~2 min per phase
- **Short SFP:** ANSEBE01 (36s), DAJARO01 (46s) — significantly shorter than expected 2 min

---

## 7. XIACT vs JSON Validation

![XIACT vs JSON Validation — 87 sessions](X:\data\Schwan_T3_FineTune\stats\xiact_vs_json_validation.png)

Post-fix, all 87 sessions show perfect alignment between source `.xiact` and converted `.json` files for both infant and caregiver tracks.

---

## 8. Output Structure

```
X:\data\Schwan_T3_FineTune\
├── icep_codes.json                          # ICEP code definitions
├── {SessionName}/
│   └── {SessionName}_finetune_annotations.json  # Enriched annotations
├── stats/
│   ├── annotation_distribution.png          # Annotation distribution charts
│   ├── annotation_stats.json                # Annotation statistics data
│   ├── session_analysis.png                 # Session coverage & SFP timeline
│   ├── session_stats.csv                    # Per-session statistics
│   ├── session_stats.json                   # Per-session statistics (JSON)
│   ├── xiact_vs_json_validation.png         # Validation results
│   ├── xiact_vs_json_validation.csv         # Validation data
│   ├── xiact_vs_json_validation.json        # Validation report
│   └── fix_log.json                         # Fix change log
```

---

## 9. Next Steps

1. **Extract audio-video chunks** — `python data_prep/01_extract_chunks.py`
2. **Create LlamaFactory dataset** — `python data_prep/02_create_dataset.py`
3. **Train/Val/Test split** — `python data_prep/03_split_data.py`
4. **Verify Qwen-Omni format** — `python data_prep/04_verify_omni_format.py`
5. **Configure LlamaFactory** — Integrate `dataset_info.json`
6. **Train** — `bash training/train_qwen_omni.sh`

### Considerations

- Address **class imbalance** (especially `Iwit` with 1 instance, `Cint`/`Cnon` with <50 each)
- Consider **trimming** the 6 low-coverage sessions to their annotated portions
- MUC vs HD site differences in play phase duration may warrant site-aware augmentation
