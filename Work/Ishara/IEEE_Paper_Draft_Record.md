# IEEE conference paper — draft record

**R26-DS-002 · Component 2 · IT22259134 · 26 August 2026**
**File:** `E:\RP\R26-DS-002\Work\Ishara\Sinhala_PostOCR_IEEE.docx` (+ `.pdf`)
**Format:** IEEE conference A4, two-column, 6 pages including 5 tables, 2 figures, 11 references.
Built on the styles of the supplied `conferencetemplatea4.docx`.

---

## 1. Framing decisions taken

| Decision | Choice |
|---|---|
| Scope | Post-OCR correction, plus a compressed Section V on the capture-side findings |
| Headline | The Sinhala benchmark and taxonomy lead; the gated negative result is Section IV-C |
| Citations | Verified against primary sources by web search; nothing invented |
| v1 CER figure (D5) | 0.0238, per `Chapter4_Results_FINAL.md` |
| Over-correction (D3) | 111 / 3,350 — the only definition under which the gated counterpart exists |
| Tesseract config | Described as what the code does: `pytesseract` defaults, `sin` traineddata |

---

## 2. Numbers verified directly from artefacts

Everything below was read out of the files under `E:\RP`, not from summary documents.

| Quantity | Value | Source |
|---|---|---|
| CER / WER, B1 / B3 / gated | 0.1197 / 0.0757 / 0.1160; 0.3358 / 0.1640 / 0.2938 | `results/final_results.json` |
| Dictionary, n-gram | CER 0.1193 / 0.1432; WER 0.3187 / 0.3517 | `results/baseline_comparison.json` |
| Recall sweep | thr 0.1–0.9, recall 0.693→0.575, gated CER 0.1158–0.1168 | `results/ablation_sweep.json` |
| Per-category | 1,329 ops; B3 61.8%, gated 22.5%; groups 698 / 364 / 267 | `results/per_category_results.json` |
| Split | 87 / 33 / 98 pages → 187 / 78 / 217 pairs; CER 0.1141 / 0.1210 / 0.1197 | `data/splits/*.json` |
| Taxonomy | 192 substitutions, 29 deletions, 92 insertions; target CER 0.1092 | `data/noise_model.json` |
| Detector | thr 0.30, recall 0.6512, precision 0.9461, FPR 0.0157 | `models/sinbert_detector/threshold.json` |
| Scraped corpus | 24,212 sentences; adaderana 22,506 (92.96%) | `data/train/scraped/corpus_v2_meta.json` |
| Training files | mT5 plain 36,743; mT5 gated 179,931; detector 36,743 | `data/train/*.json` |
| Dictionary size | 42,731 types / 448,387 tokens | recomputed from `corpus_v2.json` |
| Scale sweep 0.25×–0.60× | 0.2495 / 0.1815 / 0.1754 / 0.1817 / 0.1907 | `results/multiview/multiview_results.json` |
| Consensus | best single 0.1754 → 0.1615, +7.90% | same file |
| 2×2 end-to-end | A 0.1257 · B 0.0649 · C 0.0848 · D 0.0365 | `results/end2end_v2/end2end_v2_table.csv` |

**Bucket distribution reproduced** by re-running the alignment code over the 230 stored OCR outputs:
553 raw pairs → CORRECTABLE 489 (88.4%), BLOAT 36 (6.5%), TRUNCATED 20 (3.6%), MISALIGNED 8 (1.4%),
SCRIPT 0; 482 usable pairs from 218 pages. Raw-OCR CER over all 482 = 0.1177.

---

## 3. Corrections applied to earlier project documents

These were wrong in `Research_Full_Reference.md` and are fixed in the paper.

1. **232 photographs, not 237.** 230 have ground truth; images 26 and 33 do not.
2. **482 pairs come from 218 pages, not 230.** Twelve of the 230 transcribed pages yielded no correctable pair.
3. **The benchmark OCR did not use `--oem 1`.** `Phase1` and `Validate_Expanded_Dataset` call
   `pytesseract.image_to_string(..., lang='sin')` at default engine and PSM. `--oem 1 --psm 6` is the
   *deployed system* config in `system/core/config.py`, a different thing.
4. **The canonical run used `num_beams=4` with NO `no_repeat_ngram_size`.** The repeat-penalty result
   (0.0847 → 0.0515) is a separate NFC-normalised experiment and is not the headline configuration.
5. **The dictionary baseline improved WER by 5.1%** (0.3358 → 0.3187) even though CER was flat.
   Reporting CER alone mischaracterises it.
6. **Guan & Greene report eight languages, not seven.** German (26.25%) and Spanish (37.33%) were
   missing from the project's table. Sinhala's 36.7% therefore sits **between Spanish and Frisian**,
   not "third of seven". The paper states the range and the Telugu comparison.
7. `train_meta.json` (13 Jul) is **stale** — it describes an 11,120-sentence generation run that predates
   the 14 Jul scrape. The training files on disk correspond to the 24,212-sentence corpus.

---

## 4. Verified references

1. Dhananjaya, Demotte, Ranathunga, Jayasena — LREC 2022, pp. 7377–7385 (SinBERT)
2. Balasooriya — M.C.S. dissertation, UCSC, 2020 (53.22% → 86.16%; 419 words, 2 pages; 70,131-entry lexicon)
3. Perera & Sumanathilaka — RANLP 2025, pp. 925–933, DOI 10.26615/978-954-452-098-4-106
4. Guan & Greene — Findings of ACL 2024, pp. 6036–6047
5. Kashid & Bhattacharyya — ICON 2024 (RoundTripOCR)
6. Wasala, Weerasinghe, Pushpananda, Liyanage, Jayalatharachchi — ICTer 3(1), pp. 11–24, 2010
7. Jayatilleke & de Silva — arXiv:2507.18264, 2025 (Surya WER 2.61%, Tesseract 5.5.0 WER 14.89%, **synthetic** images)
8. Xue et al. — NAACL-HLT 2021, pp. 483–498 (mT5)
9. Smith — ICDAR 2007, pp. 629–633 (Tesseract)
10. Fiscus — ASRU 1997, pp. 347–354 (ROVER)
11. Efron — Ann. Statist. 7(1), pp. 1–26, 1979 (bootstrap)

---

## 5. Still to fill before submission

- **Authors, affiliation, emails** — marked `[INFORMATION REQUIRED]` on page 1.
- **Venue** — not stated in the draft.
- **Tesseract version** — marked `[INFORMATION REQUIRED]` in Section III-A.
- **Table V rows at 1.00×, 2.00× and 3.00×** — marked `[VERIFY]` in the table note. These are not in
  `results/multiview/multiview_results.json`; check them against `Pipeline_v11_Optimal_Capture.ipynb`
  in Drive, along with the identity of the 1.00× baseline image set.
- **The 38 → 230 page scaling figure** (0.0763 → 0.0757) is quoted from the project reference document;
  no artefact for it exists under `E:\RP`.
- **The mixed-quality consensus result** (−3.7%) is likewise from the reference document only.
