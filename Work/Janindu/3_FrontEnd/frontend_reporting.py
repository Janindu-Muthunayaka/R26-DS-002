"""
frontend_reporting.py  —  Master Report Generator for 3_FrontEnd (v2: Detailed Walkthrough)
==========================================================================================
Generates an upgraded master report page (report.html) inside the Outputs/ directory.
Each row now includes a "Preprocessing Walkthrough" tab showing the intermediate stages:
1. Layout Detection (Annotated Original)
2. Crops (Binarized Blocks)
3. OCR Results (Sentence Strips + Extracted Text)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime


def _esc(s: str) -> str:
    return (s or "")                           \
        .replace("&", "&amp;")                 \
        .replace("<", "&lt;")                  \
        .replace(">", "&gt;")                  \
        .replace('"', "&quot;")


def build_report(outputs_dir: Path, image_names: list[str]) -> Path:
    """Read per-image frontend_summary.json files and emit report.html."""

    summaries: list[dict] = []
    for name in image_names:
        stem = Path(name).stem
        sp   = outputs_dir / stem / "frontend_summary.json"
        if sp.exists():
            with open(sp, encoding="utf-8") as f:
                summaries.append(json.load(f))
        else:
            summaries.append({
                "stem":           stem,
                "fname":          name,
                "predicted_text": "",
                "texts":          [],
                "splits":         [],
                "original_image": "",
                "error":          "Summary not found",
            })

    # ── Build HTML rows ───────────────────────────────────────────────────────
    rows_html = ""
    for i, s in enumerate(summaries):
        stem          = _esc(s.get("stem", ""))
        fname         = _esc(s.get("fname", ""))
        combined_text = _esc(s.get("predicted_text", ""))
        texts         = s.get("texts", [])
        splits        = s.get("splits", [])
        orig_img      = s.get("original_image", "")
        layout_img    = s.get("layout_image", "")
        crop_imgs     = s.get("crop_images", [])
        err           = s.get("error", "")

        status_badge = ""
        if err:
            status_badge = f'<span class="badge badge-err">ERROR</span>'
        elif combined_text:
            status_badge = f'<span class="badge badge-ok">OK</span>'
        else:
            status_badge = f'<span class="badge badge-warn">NO TEXT</span>'

        # ── OCR Tab Content ───────────────────────────────────────────────────
        text_lines_html = ""
        if texts:
            for ti, t in enumerate(texts, 1):
                text_lines_html += f'<div class="text-line"><span class="text-label">Segment {ti}:</span> <span class="si-text">{_esc(t)}</span></div>'
        elif combined_text:
            text_lines_html = f'<div class="text-line"><span class="text-label">Full Text:</span> <span class="si-text">{combined_text}</span></div>'
        else:
            text_lines_html = f'<div class="text-line no-text">No text extracted</div>'

        if err:
            text_lines_html += f'<div class="error-note">FAILED {_esc(err)}</div>'

        strip_cards_html = ""
        for sp in splits:
            strip_img  = sp.get("strip_image", "")
            strip_name = _esc(sp.get("strip_name", ""))
            strip_text = _esc(sp.get("predicted_text", ""))
            detail_url = sp.get("detail_html", "")
            sp_err     = sp.get("error")

            card_inner = ""
            if strip_img:
                card_inner += f'<img src="../{strip_img}" alt="{strip_name}" class="strip-img">'
            else:
                card_inner += '<div class="strip-no-img">No image</div>'

            card_inner += f'<div class="strip-text si-text">{strip_text or "—"}</div>'
            
            stats_html = ""
            cer = sp.get("cer", 0)
            wer = sp.get("wer", 0)
            if strip_text:
                stats_html = f'<div class="strip-stats">CER: {cer:.1f}% | WER: {wer:.1f}%</div>'
            
            card_inner += stats_html
            card_inner += f'<div class="strip-name">{strip_name}</div>'
            if sp_err:
                card_inner += f'<div class="strip-err">FAILED {_esc(sp_err)}</div>'

            if detail_url:
                strip_cards_html += f'<a href="../{detail_url}" target="_blank" class="strip-card">{card_inner}</a>'
            else:
                strip_cards_html += f'<div class="strip-card">{card_inner}</div>'

        # ── Preprocessing Tab Content ──────────────────────────────────────────
        # Stage 1: Layout
        layout_html = ""
        if layout_img:
            layout_html = f"""
            <div class="walk-stage">
                <div class="walk-label">Stage 1: Layout Detection</div>
                <img src="../{layout_img}" class="walk-img layout-img" onclick="openModal('../{layout_img}')">
                <div class="walk-desc">Detected text regions (black outlines) on original image.</div>
            </div>"""
        
        # Stage 2: Crops
        crops_grid_html = ""
        if crop_imgs:
            crops_inner = "".join([f'<img src="../{c}" class="walk-crop" onclick="openModal(\'../{c}\')">' for c in crop_imgs])
            crops_grid_html = f"""
            <div class="walk-stage">
                <div class="walk-label">Stage 2: Binarized Crops</div>
                <div class="walk-crops-grid">{crops_inner}</div>
                <div class="walk-desc">Individual regions straightened and converted to black-on-white.</div>
            </div>"""

        # Stage 3: Strips (reuse splits)
        strips_inner = ""
        if splits:
            strips_inner = "".join([f'<img src="../{sp.get("strip_image")}" class="walk-strip" onclick="openModal(\'../{sp.get("strip_image")}\')">' for sp in splits if sp.get("strip_image")])
        
        if not strips_inner:
            strips_inner = '<div class="no-strips">No sentence strips found for this image.</div>'

        strips_walk_html = f"""
        <div class="walk-stage">
            <div class="walk-label">Stage 3: Sentence Strips</div>
            <div class="walk-strips-grid">{strips_inner}</div>
            <div class="walk-desc">Horizontal strips (512px height) used for OCR inference.</div>
        </div>"""

        # ── Orig image ──────────────────────────────────────────────────────────
        orig_html = ""
        if orig_img:
            orig_html = f'<img src="../{orig_img}" alt="{fname}" class="orig-thumb" onclick="openModal(\'../{orig_img}\')">'

        rows_html += f"""
  <div class="report-row" id="row-{i}">
    <div class="row-header" onclick="toggleRow({i})" role="button" aria-expanded="false" aria-controls="detail-{i}">
      <div class="row-left">
        {orig_html}
        <div class="row-meta">
          <div class="row-fname">{fname} {status_badge}</div>
          <div class="row-preview si-text">{combined_text[:80] + ('...' if len(combined_text) > 80 else '') if combined_text else '<em style="color:#555">No text</em>'}</div>
          <div class="row-stats">{len(splits)} strip(s) · {len(texts)} extract(s)</div>
        </div>
      </div>
      <div class="row-toggle">▾</div>
    </div>
    <div class="row-detail" id="detail-{i}" style="display:none;">
      <div class="row-tabs">
        <button class="tab-btn active" onclick="showTab({i}, 'ocr')">OCR Results</button>
        <button class="tab-btn" onclick="showTab({i}, 'walk')">Preprocessing Walkthrough</button>
      </div>
      
      <div id="ocr-{i}" class="tab-content active">
        <div class="detail-texts">{text_lines_html}</div>
        <div class="strip-grid">{strip_cards_html if strip_cards_html else '<div class="no-strips">No strips processed</div>'}</div>
      </div>
      
      <div id="walk-{i}" class="tab-content">
        <div class="walkthrough-layout">
          {layout_html}
          {crops_grid_html}
          {strips_walk_html}
        </div>
      </div>
    </div>
  </div>"""

    # ── Whole page ────────────────────────────────────────────────────────────
    total     = len(summaries)
    ok_count  = sum(1 for s in summaries if s.get("predicted_text"))
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="si">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sinhala OCR — Detailed Batch Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --bg:       #08090f;
  --surface:  #0f111a;
  --surface2: #151929;
  --border:   #1d2340;
  --accent:   #5e7bff;
  --green:    #4ecca3;
  --red:      #e94560;
  --amber:    #f0a500;
  --text:     #d4d9ef;
  --muted:    #5a617a;
}}
body {{
  font-family: 'Inter', sans-serif;
  background: var(--bg);
  color: var(--text);
  padding: 0;
  min-height: 100vh;
}}
header {{
  background: linear-gradient(135deg, #0d1035 0%, #1a2060 100%);
  padding: 40px 60px;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
header .titles h1 {{
  font-size: 28px;
  font-weight: 800;
  color: #fff;
  letter-spacing: -0.5px;
  margin-bottom: 6px;
}}
header .subtitle {{
  font-size: 14px;
  color: var(--muted);
}}
.stats-bar {{
  display: flex;
  gap: 32px;
  padding: 24px 60px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
}}
.stat {{
  display: flex;
  flex-direction: column;
  gap: 4px;
}}
.stat .val {{
  font-size: 32px;
  font-weight: 800;
  color: var(--accent);
}}
.stat .lbl {{
  font-size: 12px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  font-weight: 600;
}}
main {{
  padding: 40px 60px;
  max-width: 1600px;
  margin: 0 auto;
}}
.report-row {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  margin-bottom: 20px;
  overflow: hidden;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}}
.report-row:hover {{ 
  border-color: var(--accent);
  box-shadow: 0 8px 32px rgba(94,123,255,0.15);
}}
.row-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 24px 28px;
  cursor: pointer;
  user-select: none;
  gap: 24px;
}}
.row-left {{
  display: flex;
  align-items: center;
  gap: 24px;
  flex: 1;
  min-width: 0;
}}
.orig-thumb {{
  width: 80px;
  height: 80px;
  object-fit: cover;
  border-radius: 12px;
  background: #fff;
  flex-shrink: 0;
  border: 1px solid var(--border);
  transition: transform 0.2s;
}}
.orig-thumb:hover {{ transform: scale(1.05); }}
.row-meta {{
  flex: 1;
  min-width: 0;
}}
.row-fname {{
  font-size: 18px;
  font-weight: 700;
  color: #fff;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 12px;
}}
.row-preview {{
  font-size: 20px;
  color: var(--green);
  margin-bottom: 6px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  letter-spacing: 1px;
}}
.row-stats {{
  font-size: 13px;
  color: var(--muted);
  font-weight: 500;
}}
.row-toggle {{
  font-size: 24px;
  color: var(--muted);
  transition: transform 0.4s;
  flex-shrink: 0;
}}
.row-header.open .row-toggle {{ transform: rotate(180deg); color: var(--accent); }}
.badge {{
  font-size: 11px;
  font-weight: 800;
  padding: 4px 10px;
  border-radius: 20px;
  letter-spacing: 0.8px;
  text-transform: uppercase;
}}
.badge-ok   {{ background: rgba(78,204,163,0.15); color: var(--green); border: 1px solid rgba(78,204,163,0.2); }}
.badge-err  {{ background: rgba(233,69,96,0.15);  color: var(--red);   border: 1px solid rgba(233,69,96,0.2); }}
.badge-warn {{ background: rgba(240,165,0,0.15);  color: var(--amber); border: 1px solid rgba(240,165,0,0.2); }}

.row-detail {{
  border-top: 1px solid var(--border);
  padding: 32px;
  background: linear-gradient(to bottom, #0f111a, #0b0d14);
  display: none;
}}
.row-tabs {{
  display: flex;
  gap: 12px;
  margin-bottom: 28px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 12px;
}}
.tab-btn {{
  background: transparent;
  border: none;
  color: var(--muted);
  font-family: inherit;
  font-size: 14px;
  font-weight: 700;
  padding: 10px 20px;
  cursor: pointer;
  border-radius: 8px;
  transition: all 0.2s;
}}
.tab-btn:hover {{ color: var(--text); background: var(--surface2); }}
.tab-btn.active {{ color: #fff; background: var(--accent); }}

.tab-content {{ display: none; }}
.tab-content.active {{ display: block; animation: fadeIn 0.3s ease-out; }}

@keyframes fadeIn {{
  from {{ opacity: 0; transform: translateY(4px); }}
  to   {{ opacity: 1; transform: translateY(0);    }}
}}

/* ── OCR Tab Styles ── */
.detail-texts {{ margin-bottom: 32px; }}
.text-line {{
  padding: 16px 20px;
  background: var(--surface2);
  border-radius: 12px;
  margin-bottom: 12px;
  display: flex;
  align-items: baseline;
  gap: 16px;
  border: 1px solid transparent;
  transition: border-color 0.2s;
}}
.text-line:hover {{ border-color: var(--border); }}
.text-label {{
  font-size: 11px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  font-weight: 700;
  width: 100px;
  flex-shrink: 0;
}}
.si-text {{
  font-size: 24px;
  color: var(--green);
  line-height: 1.5;
  letter-spacing: 2.5px;
  font-weight: 400;
}}
.strip-grid {{ display: flex; flex-wrap: wrap; gap: 20px; }}
.strip-card {{
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px;
  width: 240px;
  text-decoration: none;
  color: inherit;
  transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
  display: block;
}}
.strip-card:hover {{
  border-color: var(--accent);
  transform: translateY(-6px);
  box-shadow: 0 12px 32px rgba(94,123,255,0.25);
}}
.strip-img {{
  width: 100%;
  height: 90px;
  object-fit: contain;
  background: #fff;
  border-radius: 8px;
  display: block;
  margin-bottom: 12px;
}}
.strip-text {{
  font-size: 20px;
  text-align: center;
  margin-bottom: 4px;
  min-height: 30px;
  letter-spacing: 1.5px;
  color: var(--green);
}}
.strip-stats {{
  font-size: 11px;
  color: var(--amber);
  text-align: center;
  margin-bottom: 8px;
  font-weight: 600;
}}
.strip-name {{ font-size: 11px; color: var(--muted); text-align: center; text-overflow: ellipsis; overflow: hidden; white-space: nowrap; }}

/* ── Walkthrough Tab Styles ── */
.walkthrough-layout {{ display: flex; flex-direction: column; gap: 40px; }}
.walk-stage {{ background: var(--surface2); border-radius: 16px; padding: 24px; border: 1px solid var(--border); }}
.walk-label {{ font-size: 14px; font-weight: 800; color: var(--accent); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 20px; border-left: 4px solid var(--accent); padding-left: 12px; }}
.walk-img {{ max-width: 100%; border-radius: 12px; display: block; border: 1px solid var(--border); cursor: zoom-in; }}
.layout-img {{ max-height: 600px; object-fit: contain; background: #fff; }}
.walk-desc {{ font-size: 13px; color: var(--muted); margin-top: 16px; font-style: italic; }}
.walk-crops-grid {{ display: flex; flex-wrap: wrap; gap: 12px; }}
.walk-crop {{ height: 120px; border-radius: 8px; border: 1px solid var(--border); background: #fff; object-fit: contain; cursor: zoom-in; transition: transform 0.2s; }}
.walk-crop:hover {{ transform: scale(1.1); z-index: 2; }}
.walk-strips-grid {{ display: flex; flex-direction: column; gap: 12px; }}
.walk-strip {{ width: 100%; height: 80px; object-fit: contain; border-radius: 8px; border: 1px solid var(--border); background: #fff; cursor: zoom-in; }}

/* ── Modal ── */
#modal {{
  display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; 
  background: rgba(0,0,0,0.9); align-items: center; justify-content: center;
}}
#modal img {{ max-width: 90%; max-height: 90%; border-radius: 8px; box-shadow: 0 0 40px rgba(0,0,0,0.5); }}
#modal:target {{ display: flex; }}

</style>
</head>
<body>
<header>
  <div class="titles">
    <h1>🔤 Sinhala OCR Detailed Report</h1>
    <div class="subtitle">Multi-stage pipeline analysis · {generated}</div>
  </div>
</header>

<div class="stats-bar">
  <div class="stat"><div class="val">{total}</div><div class="lbl">Images</div></div>
  <div class="stat"><div class="val">{ok_count}</div><div class="lbl">Successful</div></div>
  <div class="stat"><div class="val">{total - ok_count}</div><div class="lbl">Issues</div></div>
</div>

<main>
{rows_html}
</main>

<div id="modal" onclick="this.style.display='none'">
  <img id="modal-img" src="" alt="">
</div>

<script>
function toggleRow(i) {{
  const detail = document.getElementById('detail-' + i);
  const header = detail.previousElementSibling;
  const isOpen = detail.style.display === 'block';
  
  if (isOpen) {{
    detail.style.display = 'none';
    header.classList.remove('open');
    header.setAttribute('aria-expanded', 'false');
  }} else {{
    detail.style.display = 'block';
    header.classList.add('open');
    header.setAttribute('aria-expanded', 'true');
  }}
}}

function showTab(rowIdx, tabType) {{
  const detail = document.getElementById('detail-' + rowIdx);
  const btns = detail.querySelectorAll('.tab-btn');
  const contents = detail.querySelectorAll('.tab-content');
  
  btns.forEach(b => b.classList.remove('active'));
  contents.forEach(c => c.classList.remove('active'));
  
  // Find the button that corresponds to the tab type
  const targetBtn = Array.from(btns).find(b => {{
    const txt = b.textContent.toLowerCase();
    if (tabType === 'ocr') return txt.includes('ocr');
    return txt.includes('walkthrough') || txt.includes('preprocess');
  }});
  
  const targetContent = document.getElementById(tabType + '-' + rowIdx);
  
  if (targetBtn) targetBtn.classList.add('active');
  if (targetContent) targetContent.classList.add('active');
}}

function openModal(src) {{
  const modal = document.getElementById('modal');
  const modalImg = document.getElementById('modal-img');
  modalImg.src = src;
  modal.style.display = 'flex';
}}
</script>
</body>
</html>"""

    out_path = outputs_dir / "report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Report] report.html -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--images",  nargs="+", required=True)
    args = ap.parse_args()

    out_dir = Path(args.outputs)
    build_report(out_dir, args.images)


if __name__ == "__main__":
    main()
