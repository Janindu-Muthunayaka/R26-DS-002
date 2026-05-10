# =============================================================================
# stage4_classification.py  -  Model Loading, GPU Inference & Recognition
#
# Handles the EfficientNetV2-S classification pipeline:
#   batched GPU inference → variant-map recognition → word annotation → Metrics
#
# INPUT  (from Stage 3 - Segmentation):
#   - stems        : list[str]             - image stems to process
#   - work_root    : str                   - base working directory
#   - model        : nn.Module             - loaded EfficientNetV2-S
#   - device       : torch.device          - CUDA or CPU
#   - idx_to_class : dict[str, str]        - {index: class_name}
#   - cfg          : PipelineConfig        - tunable parameters
#
# OUTPUT (to stage5_reporting):
#   - list[dict]  - per-image result dicts
#
# Recognition logic - Variant-Map Recognition (replaces greedy multi-seg):
# ─────────────────────────────────────────────────────────────────────────
#   For each segment position `pos` in the sentence:
#
#   1. Predict the single-segment crop at `pos`  → base_char, base_conf
#
#   2. Determine word boundary context from Stage 3 word groups:
#        - has_left  : pos-1 exists AND is in the same word
#        - has_right : pos+1 exists AND is in the same word
#        (Special 2-back rule: only ම and ව trigger look-back of 2 segments)
#
#   3. Build candidate crops (respecting word boundaries + overlap rules):
#        - crop_left       : segments[pos-1..pos]   (if has_left)
#        - crop_right      : segments[pos..pos+1]   (if has_right)
#        - crop_both       : segments[pos-1..pos+1] (if has_left AND has_right)
#
#   4. Batch-predict all non-None candidate crops.
#
#   5. Validate candidates against VARIANT_MAP:
#        For each candidate crop, its top-1 predicted class must appear as a
#        VALUE under the KEY that equals `base_char` in VARIANT_MAP.
#        Candidates that are not in VARIANT_MAP[base_char] are discarded.
#
#   6. Priority among validated candidates:
#        a. If base_char itself is a non-key (already a full akshara with
#           diacritics - i.e. NOT a bare consonant key in VARIANT_MAP):
#              → HIGHEST priority; still test left/right for completeness
#                but this standalone prediction wins if its confidence ≥
#                all validated compound candidates.
#        b. Among compound candidates: both > right > left
#        c. If confidence tie: both > right > left (priority breaks tie)
#   Uses SegmentLogic.py to process raw model tokens into valid characters.
# =============================================================================

from __future__ import annotations

import os
import json
import concurrent.futures
import csv
import sys
from typing import Optional

import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision import models
from torch.amp import autocast

from stage1_config import (
    PipelineConfig,
    _NORM_MEAN, _NORM_STD,
    INFER_BATCH_SIZE, P_CHAR_CANVAS_SIZE, TOP_K, PNG_POOL_WORKERS,
)
from stage3_segmentation import _make_window_crop_np

_INFO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Information")
if _INFO_DIR not in sys.path:
    sys.path.insert(0, _INFO_DIR)

import SegmentLogic as seg_logic

# =============================================================================
# CLASS MAP
# =============================================================================
_CLASS_MAP = {}
def _class_to_sinhala(class_name: str) -> str:
    """Extract the Sinhala character from a class name like 'vowel_අ_0001'."""
    global _CLASS_MAP
    if not _CLASS_MAP:
        path = os.path.join(_INFO_DIR, "class_map.csv")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    _CLASS_MAP[row["name"]] = row["symbol"]
    
    parts = class_name.split("_")
    fallback = parts[1] if len(parts) >= 3 else class_name
    return _CLASS_MAP.get(class_name, fallback)

# =============================================================================
# PNG WRITE POOL  (I/O off the main thread)
# =============================================================================

_PNG_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=PNG_POOL_WORKERS)

def _write_png_blocking(numpy_img: np.ndarray, path: str) -> None:
    _, buf = cv2.imencode(".png", numpy_img)
    with open(path, "wb") as fh:
        fh.write(buf)

def _save_png_async(numpy_img: np.ndarray,
                    path: str) -> concurrent.futures.Future:
    return _PNG_POOL.submit(_write_png_blocking, numpy_img.copy(), path)

# =============================================================================
# MODEL LOADING
# =============================================================================

def _load_model(model_path: str, num_classes: int, device: torch.device):
    m = models.efficientnet_v2_s(weights=None)
    in_features       = m.classifier[1].in_features
    m.classifier[1]   = nn.Linear(in_features, num_classes)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    m.load_state_dict(ckpt["model_state_dict"])
    m.to(device).eval()

    val_acc = ckpt.get("val_acc", 0)
    if isinstance(val_acc, torch.Tensor):
        val_acc = val_acc.item()

    print(f"  [Stage 4 - Classification] Model     : EfficientNetV2-S  ({num_classes} classes)")
    print(f"  [Stage 4 - Classification] Device    : {device}")
    print(f"  [Stage 4 - Classification] Checkpoint: epoch {ckpt.get('epoch', '?')} | "
          f"val acc {val_acc * 100:.2f}%")

    if device.type == "cuda":
        _warmup_model(m, device, num_classes)
    return m

def _warmup_model(model, device: torch.device, num_classes: int) -> None:
    cs = P_CHAR_CANVAS_SIZE
    try:
        dummy = torch.zeros(1, 3, cs, cs, device=device)
        with torch.no_grad():
            with autocast(device_type="cuda"):
                _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"  [Stage 4 - Classification] cuDNN warmup complete  (batch size = 1, {cs}×{cs})")
    except Exception as exc:
        print(f"  [Stage 4 - Classification] cuDNN warmup skipped: {exc}")

# =============================================================================
# NUMPY → PINNED TENSOR
# =============================================================================

def _np_to_tensor_pinned(gray_np: np.ndarray,
                          device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(gray_np).float().div_(255.0)
    t = t.unsqueeze(0).expand(3, -1, -1).contiguous()
    if device.type == "cuda":
        t = t.pin_memory()
        t = t.to(device, non_blocking=True)
    else:
        t = t.to(device)
    mean = _NORM_MEAN.to(device)
    std  = _NORM_STD.to(device)
    t    = t.sub_(mean[:, None, None]).div_(std[:, None, None])
    return t.unsqueeze(0)

# =============================================================================
# BATCHED INFERENCE
# =============================================================================

def _predict_batch(crops_np: list,
                   model,
                   device: torch.device,
                   idx_to_class: dict) -> list:
    valid_idx = [i for i, c in enumerate(crops_np) if c is not None]
    if not valid_idx:
        return [None] * len(crops_np)
    all_preds = [None] * len(crops_np)
    for chunk_start in range(0, len(valid_idx), INFER_BATCH_SIZE):
        chunk = valid_idx[chunk_start: chunk_start + INFER_BATCH_SIZE]
        tensors = torch.cat(
            [_np_to_tensor_pinned(crops_np[i], device) for i in chunk], dim=0
        )
        with torch.no_grad():
            if device.type == "cuda":
                with autocast(device_type="cuda"):
                    logits = model(tensors)
                    probs_all = torch.softmax(logits.float(), dim=1)
            else:
                logits    = model(tensors)
                probs_all = torch.softmax(logits, dim=1)
            top_probs, top_i = torch.topk(probs_all, k=TOP_K)
        top_probs_cpu = top_probs.cpu()
        top_i_cpu     = top_i.cpu()
        for batch_pos, orig_idx in enumerate(chunk):
            all_preds[orig_idx] = [
                (idx_to_class[str(top_i_cpu[batch_pos, k].item())],
                 round(top_probs_cpu[batch_pos, k].item() * 100, 2))
                for k in range(TOP_K)
            ]
    return all_preds

# =============================================================================
# WORD-BOUNDARY HELPER
# =============================================================================

def _build_seg_to_word(word_groups: list) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for wg in word_groups:
        for si in wg["seg_indices"]:
            mapping[si] = wg["word_index"]
    return mapping

# =============================================================================
# FSM RECOGNITION
# =============================================================================

def _fsm_segment(skeleton:    np.ndarray,
                  segments:   list,
                  word_groups: list,
                  model,
                  device:      torch.device,
                  idx_to_class: dict,
                  out_dir:     str,
                  cfg:         PipelineConfig) -> list:
    """
    Buffered Finite-State-Machine recognition loop using SegmentLogic.py.
    """
    N = len(segments)
    seg_to_word = _build_seg_to_word(word_groups) if word_groups else {}
    png_futures = []

    # ── 1. Predict all individual segments ───────────────────────────────────
    raw_tokens = []
    for pos, seg in enumerate(segments):
        crop = _make_window_crop_np(skeleton, seg[0], seg[1], cfg)
        if crop is None:
            raw_tokens.append(None)
            continue
            
        [preds] = _predict_batch([crop], model, device, idx_to_class)
        top_class = preds[0][0]
        top_conf = preds[0][1]
        
        parts = top_class.split('_')
        ctype = parts[0].lower() if len(parts) >= 3 else "unknown"
        if ctype == "just": ctype = "noise"
            
        char = _class_to_sinhala(top_class)
        
        tok_crop_path = os.path.join(out_dir, f"token_{pos:03d}.png")
        png_futures.append(_save_png_async(crop, tok_crop_path))
        
        raw_tokens.append({
            "pos": pos,
            "seg": seg,
            "crop": crop,
            "crop_path": tok_crop_path,
            "preds": preds,
            "top_class": top_class,
            "class_type": ctype,
            "char": char,
            "conf": top_conf
        })

    # ── 2. Process through SegmentLogic ──────────────────────────────────────
    output_buffer = []
    
    pos = 0
    loop_guard = 0
    while pos < N:
        loop_guard += 1
        if loop_guard > 5000:
            print(f"      [EMERGENCY BREAK] Infinite loop detected in FSM for image at pos {pos}. Skipping image.")
            break
            
        tok = raw_tokens[pos]
        if tok is None:
            pos += 1
            continue
            
        char = tok["char"]
        ctype = tok["class_type"]
        
        if ctype == "noise":
            pos += 1
            continue
            
        if ctype in ("combo", "hal", "vowel", "punct", "unknown"):
            output_buffer.append({"final_char": char, "tokens": [tok], "fsm_inputs": None})
            pos += 1
            continue
            
        if ctype == "pili":
            try:
                CurrentPos = char
                
                CurrentNext_tok = None
                for i in range(pos + 1, N):
                    if raw_tokens[i] and raw_tokens[i]["class_type"] != "noise":
                        CurrentNext_tok = raw_tokens[i]
                        break
                CurrentNext = CurrentNext_tok["char"] if CurrentNext_tok else None
                
                NextNext_tok = None
                if CurrentNext_tok:
                    for i in range(CurrentNext_tok["pos"] + 1, N):
                        if raw_tokens[i] and raw_tokens[i]["class_type"] != "noise":
                            NextNext_tok = raw_tokens[i]
                            break
                NextNext = NextNext_tok["char"] if NextNext_tok else None
                
                Prev_dict = output_buffer[-1] if len(output_buffer) >= 1 else None
                PrevPrev_dict = output_buffer[-2] if len(output_buffer) >= 2 else None
                
                Prev = Prev_dict["final_char"] if Prev_dict else None
                PrevPrev = PrevPrev_dict["final_char"] if PrevPrev_dict else None
                
                string_buffer = [b["final_char"] for b in output_buffer]
                
                final_class, consumed = seg_logic.process_pili_trigger(
                    CurrentPos, CurrentNext, NextNext, Prev, PrevPrev, string_buffer
                )
                
                fsm_inputs = {
                    "PrevPrev": PrevPrev,
                    "Prev": Prev,
                    "CurrentPos": CurrentPos,
                    "CurrentNext": CurrentNext,
                    "NextNext": NextNext
                }
                
                pops_needed = len(output_buffer) - len(string_buffer)
                popped_tokens = []
                for _ in range(pops_needed):
                    popped_dict = output_buffer.pop()
                    popped_tokens = popped_dict["tokens"] + popped_tokens
                    
                if not final_class.startswith("[NOISE"):
                    consumed_tokens = []
                    curr_i = pos + 1
                    while len(consumed_tokens) < consumed and curr_i < N:
                        if raw_tokens[curr_i] and raw_tokens[curr_i]["class_type"] != "noise":
                            consumed_tokens.append(raw_tokens[curr_i])
                        curr_i += 1
                        
                    merged_tokens = popped_tokens + [tok] + consumed_tokens
                    output_buffer.append({"final_char": final_class, "tokens": merged_tokens, "fsm_inputs": fsm_inputs})
                    
                    if consumed > 0 and consumed_tokens:
                        pos = consumed_tokens[-1]["pos"] + 1
                    else:
                        pos += 1
                else:
                    pos += 1
            except Exception as e:
                print(f"      [FSM ERROR] Error at pos {pos}: {e}")
                pos += 1
            continue

    # ── 3. Format into final results array ───────────────────────────────────
    results = []
    akshara_idx = 0
    
    for item in output_buffer:
        tokens = item["tokens"]
        if not tokens: continue
        
        final_char = item["final_char"]
        seg_start_idx = min(t["pos"] for t in tokens)
        seg_end_idx = max(t["pos"] for t in tokens)
        x_start = min(t["seg"][0] for t in tokens)
        x_end = max(t["seg"][1] for t in tokens)
        
        w_crop = _make_window_crop_np(skeleton, x_start, x_end, cfg)
        if w_crop is None: continue
        
        crop_path = os.path.join(out_dir, f"akshara_{akshara_idx:03d}.png")
        png_futures.append(_save_png_async(w_crop, crop_path))
        
        primary_tok = tokens[0]
        for t in tokens:
            if t["class_type"] in ("combo", "vowel", "hal"):
                primary_tok = t
                break
                
        results.append({
            "index":          akshara_idx,
            "seg_start":      seg_start_idx,
            "seg_end":        seg_end_idx,
            "window_segs":    len(tokens),
            "x_start":        x_start,
            "x_end":          x_end,
            "chosen_by":      "SegmentLogic" if len(tokens) > 1 else "Base",
            "confidence":     primary_tok["conf"],
            "crop_path":      crop_path,
            "predictions":    [[p[0], p[1]] for p in primary_tok["preds"]],
            "predicted_char": final_char,
            "word_index":     seg_to_word.get(seg_start_idx, 0) if seg_to_word else 0,
            "debug_trace": {
                "fsm_rule": "SegmentLogic" if len(tokens) > 1 else "Base",
                "fsm_inputs": item.get("fsm_inputs"),
                "raw_tokens": [
                    {
                        "pos": t["pos"],
                        "char": t["char"],
                        "class_type": t["class_type"],
                        "conf": t["conf"],
                        "top_class": t["top_class"],
                        "crop_path": t.get("crop_path", ""),
                        "predictions": [[p[0], p[1]] for p in t["preds"][:3]]
                    } for t in tokens
                ]
            }
        })
        akshara_idx += 1

    concurrent.futures.wait(png_futures)
    return results

def _greedy_segment(skeleton, segments, model, device, idx_to_class,
                    out_dir, cfg, word_groups=None):
    wg       = word_groups or []
    return _fsm_segment(skeleton, segments, wg, model, device,
                                 idx_to_class, out_dir, cfg)

# =============================================================================
# WORD ANNOTATION & METRICS
# =============================================================================

def _annotate_word_indices(char_results: list, word_groups: list) -> str:
    if not word_groups:
        for ak in char_results: ak["word_index"] = 0
        return "".join(ak["predicted_char"] for ak in char_results)
    seg_to_word = {}
    for wg in word_groups:
        for si in wg["seg_indices"]: seg_to_word[si] = wg["word_index"]
    for ak in char_results:
        ak["word_index"] = seg_to_word.get(ak["seg_start"], 0)
    words_text         = []
    prev_word          = None
    current_word_chars = []
    for ak in char_results:
        wi = ak["word_index"]
        if prev_word is None: prev_word = wi
        if wi != prev_word:
            words_text.append("".join(current_word_chars))
            current_word_chars = []
            prev_word = wi
        current_word_chars.append(ak["predicted_char"])
    if current_word_chars: words_text.append("".join(current_word_chars))
    return " ".join(words_text)

def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp   = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp  = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev  = temp
    return dp[n]

def compute_cer(pred: str, ref: str) -> float:
    if not ref: return 0.0
    return round(_edit_distance(list(pred), list(ref)) / len(list(ref)) * 100, 2)

def compute_wer(pred: str, ref: str) -> float:
    pw, rw = pred.split(), ref.split()
    if not rw: return 0.0
    return round(_edit_distance(pw, rw) / len(rw) * 100, 2)

# =============================================================================
# PROCESS ONE IMAGE
# =============================================================================

def _process_one(stem: str,
                 work_root: str,
                 model,
                 device: torch.device,
                 idx_to_class: dict,
                 cfg: PipelineConfig) -> dict:
    temp_dir  = os.path.join(work_root, "temp", stem)
    meta_path = os.path.join(temp_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f: meta = json.load(f)

    ground_truth  = meta["ground_truth"]
    skeleton_path = meta.get("skel_refined_path") or meta.get("skel_path")
    skeleton      = cv2.bitwise_not(cv2.imread(skeleton_path, cv2.IMREAD_GRAYSCALE))

    with open(meta["segments_path"], "r", encoding="utf-8") as f:
        segments = json.load(f)

    N = len(segments)
    if N == 0:
        results_data = {
            "stem": stem, "ground_truth": ground_truth,
            "predicted_text": "", "predicted_text_no_spaces": "",
            "wer": 100.0, "cer": 100.0,
            "word_spacer_enabled": cfg.word_spacer_enabled, "aksharas": [],
        }
    else:
        # ── Load word groups (always, so variant-map respects word boundaries) ─
        word_groups: list = []
        if cfg.word_spacer_enabled and meta.get("word_segments_path"):
            ws_path = meta["word_segments_path"]
            if os.path.exists(ws_path):
                with open(ws_path, "r", encoding="utf-8") as f:
                    word_groups = json.load(f)

        # ── Run FSM recognition ───────────────────────────────────────────────
        char_results = _fsm_segment(
            skeleton     = skeleton,
            segments     = segments,
            word_groups  = word_groups,
            model        = model,
            device       = device,
            idx_to_class = idx_to_class,
            out_dir      = temp_dir,
            cfg          = cfg,
        )

        predicted_text_plain = "".join(r["predicted_char"] for r in char_results)

        # ── Word annotation ───────────────────────────────────────────────────
        if cfg.word_spacer_enabled and word_groups:
            predicted_text = _annotate_word_indices(char_results, word_groups)
        else:
            predicted_text = predicted_text_plain
            for ak in char_results: ak["word_index"] = 0

        wer     = compute_wer(predicted_text, ground_truth) if ground_truth else 0.0
        cer     = compute_cer(predicted_text, ground_truth) if ground_truth else 0.0
        n_words = (max((ak["word_index"] or 0) for ak in char_results) + 1
                   if char_results else 0)

        results_data = {
            "stem":                    stem,
            "ground_truth":            ground_truth,
            "predicted_text":          predicted_text,
            "predicted_text_no_spaces": predicted_text_plain,
            "wer":                     wer,
            "cer":                     cer,
            "n_segments":              N,
            "n_aksharas":              len(char_results),
            "n_multi_seg":             sum(1 for r in char_results
                                          if r["window_segs"] > 1),
            "n_words":                 n_words if cfg.word_spacer_enabled else None,
            "multi_seg_threshold":     cfg.multi_seg_threshold,
            "word_spacer_enabled":     cfg.word_spacer_enabled,
            "aksharas":                char_results,
        }

    with open(os.path.join(temp_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    return results_data

# =============================================================================
# RUN STAGE 4 - CLASSIFICATION
# =============================================================================

def run_stage4_classification(cfg: PipelineConfig,
                               stems: list,
                               work_root: str,
                               model,
                               device: torch.device,
                               idx_to_class: dict) -> list:
    print(f"  [Stage 4 - Classification] Classifying {len(stems)} image(s) on {device}...")
    all_results = []
    for i, stem in enumerate(stems, 1):
        print(f"    [{i}/{len(stems)}]", end=" ")
        try:
            res = _process_one(stem, work_root, model, device, idx_to_class, cfg)
            all_results.append(res)
            print()
        except Exception as exc:
            import traceback
            print(f"\n           ERROR: {exc}")
            traceback.print_exc()
    if device.type == "cuda": torch.cuda.synchronize()
    print(f"  [Stage 4 - Classification] Done.\n")
    return all_results