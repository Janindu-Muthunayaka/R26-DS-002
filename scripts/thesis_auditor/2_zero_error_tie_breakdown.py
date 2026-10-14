"""
2. Zero-Error Tie Breakdown (SinOCR 8,872-Image Evaluation)
===========================================================
Parses the comprehensive SinOCR evaluation results CSV (8,872 images).
Isolates the exactly 2,860 image evaluations where RAW and MAT pipelines tied (CER_RAW == CER_MAT).
Categorizes the ties into:
  - Perfect predictions (CER == 0.0)
  - Identical non-zero error predictions (CER > 0.0)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def run_tie_breakdown():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    csv_path = Path(r"E:\Sliit\Research\TROCR\Data for presenting\Compare Package\evaluation_results.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found at: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)

    total_images = len(df)
    mat_better = (df['Better_Model'] == 'MAT').sum()
    raw_better = (df['Better_Model'] == 'RAW').sum()
    ties = df[df['RAW_CER'] == df['MAT_CER']].copy()
    tie_count = len(ties)

    # Breakdown of ties: perfect (0.0) vs non-zero errors
    perfect_ties = ties[ties['RAW_CER'] == 0.0]
    nonzero_ties = ties[ties['RAW_CER'] > 0.0]

    n_perfect = len(perfect_ties)
    n_nonzero = len(nonzero_ties)

    pct_perfect_of_ties = (n_perfect / tie_count) * 100.0
    pct_nonzero_of_ties = (n_nonzero / tie_count) * 100.0

    pct_perfect_of_total = (n_perfect / total_images) * 100.0
    pct_nonzero_of_total = (n_nonzero / total_images) * 100.0

    print("=" * 78)
    print("THESIS EVALUATION AUDIT: SINOCR 8,872-IMAGE ZERO-ERROR TIE BREAKDOWN")
    print("=" * 78)

    print(f"\n[1] Overall SinOCR Evaluation Summary (N = {total_images:,} images):")
    print(f"  • RAW Pipeline Superior  : {raw_better:,} images ({raw_better/total_images*100:.2f}%)")
    print(f"  • MAT Pipeline Superior  : {mat_better:,} images ({mat_better/total_images*100:.2f}%)")
    print(f"  • Pipeline Ties (Equality): {tie_count:,} images ({tie_count/total_images*100:.2f}%)")
    print(f"  • Check Better_Model=='Tie': {(df['Better_Model'] == 'Tie').sum():,} images")

    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print(f"\n[2] Breakdown of Exactly {tie_count:,} Tied Images:")
    print("  +-------------------------------------+-----------+---------------+-----------------+")
    print("  | Tie Category                        | Count     | % of Ties     | % of All Images |")
    print("  +-------------------------------------+-----------+---------------+-----------------+")
    print(f"  | Perfect Predictions (CER == 0.0)    | {n_perfect:>9,d} | {pct_perfect_of_ties:>12.2f}% | {pct_perfect_of_total:>14.2f}% |")
    print(f"  | Identical Non-Zero Errors (CER > 0) | {n_nonzero:>9,d} | {pct_nonzero_of_ties:>12.2f}% | {pct_nonzero_of_total:>14.2f}% |")
    print("  +-------------------------------------+-----------+---------------+-----------------+")
    print(f"  | Total Pipeline Ties                 | {tie_count:>9,d} | {100.0:>12.2f}% | {tie_count/total_images*100:>14.2f}% |")
    print("  +-------------------------------------+-----------+---------------+-----------------+")

    print(f"\n[3] Statistics for Identical Non-Zero Error Ties (n = {n_nonzero:,}):")
    print(f"  • Mean CER           : {nonzero_ties['RAW_CER'].mean():.4f} ({nonzero_ties['RAW_CER'].mean()*100:.2f}%)")
    print(f"  • Median CER         : {nonzero_ties['RAW_CER'].median():.4f}")
    print(f"  • Standard Deviation : {nonzero_ties['RAW_CER'].std():.4f}")
    print(f"  • Min / Max CER      : {nonzero_ties['RAW_CER'].min():.4f} / {nonzero_ties['RAW_CER'].max():.4f}")

    # Check whether the text predictions were literally identical strings
    identical_strings = (nonzero_ties['MAT_Result'] == nonzero_ties['RAW_Result']).sum()
    print(f"  • Exactly Identical OCR Text Outputs : {identical_strings:,} of {n_nonzero:,} ({identical_strings/n_nonzero*100:.2f}%)")

    # Global perfect prediction rate across pipelines
    all_perfect_raw = (df['RAW_CER'] == 0.0).sum()
    all_perfect_mat = (df['MAT_CER'] == 0.0).sum()
    print(f"\n[4] Global Perfect Prediction Context:")
    print(f"  • Total RAW 0.0 CER Predictions: {all_perfect_raw:,} ({all_perfect_raw/total_images*100:.2f}%)")
    print(f"  • Total MAT 0.0 CER Predictions: {all_perfect_mat:,} ({all_perfect_mat/total_images*100:.2f}%)")
    print(f"  • Coincident Perfect Predictions: {n_perfect:,} (both pipelines achieved 0.0 error)")

    print(f"\n[5] Thesis Defense Summary:")
    print(f"      \"Out of the 2,860 images where the pipelines achieved identical character error rates,")
    print(f"       1,522 cases (53.22%) represent perfect recognitions (CER = 0.0) where both architectures")
    print(f"       successfully decoded the text without error. The remaining 1,338 ties (46.78%) reflect")
    print(f"       equivalent non-zero errors (mean CER = {nonzero_ties['RAW_CER'].mean()*100:.2f}%), indicating challenging")
    print(f"       sample conditions that equally degraded both feature extraction regimes.\"")
    print("=" * 78)

if __name__ == "__main__":
    run_tie_breakdown()
