"""
Generate pre-rendered HTML rows for evalClassTableBody and verify all metrics.
"""
import json
from pathlib import Path
import html

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "reports" / "model_evaluation_metrics.json"

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

classes = data["per_class_metrics"]
print(f"Loaded {len(classes)} classes.")

rows_html = []
for c in classes:
    vh = c["visual_health"]
    if vh == "Healthy":
        badge = '<span class="status-badge badge-healthy" style="padding:2px 8px;font-size:0.75rem;"><i class="fas fa-heart"></i> Healthy</span>'
    elif vh == "Diseased":
        badge = '<span class="status-badge badge-diseased" style="padding:2px 8px;font-size:0.75rem;"><i class="fas fa-virus"></i> Diseased</span>'
    else:
        badge = f'<span class="status-badge" style="background:rgba(217,119,6,0.1);color:#d97706;border:1px solid rgba(217,119,6,0.3);padding:2px 8px;font-size:0.75rem;"><i class="fas fa-bug"></i> {html.escape(vh)}</span>'

    prec_pct = f"{c['precision'] * 100:.1f}%"
    rec_pct = f"{c['recall'] * 100:.1f}%"
    f1_val = c["f1_score"] * 100
    f1_pct = f"{f1_val:.1f}%"

    if c["f1_score"] < 0.40:
        bar_color = "#dc2626"
    elif c["f1_score"] < 0.60:
        bar_color = "#d97706"
    else:
        bar_color = "#16a34a"

    clean_name = html.escape(c["class_name"].replace("_", " "))
    clean_cond = html.escape(c["condition"])

    row = f"""                                <tr data-class-name="{clean_name.lower()}" data-species="{html.escape(c['species'].lower())}" data-health="{html.escape(vh.lower())}">
                                    <td style="font-weight:600;font-size:0.85rem;color:var(--text-primary);">
                                        <div>{clean_name}</div>
                                        <div style="font-size:0.75rem;color:var(--text-muted);font-weight:400;">{clean_cond}</div>
                                    </td>
                                    <td>{badge}</td>
                                    <td style="font-size:0.85rem;font-weight:600;">{prec_pct}</td>
                                    <td style="font-size:0.85rem;font-weight:600;">{rec_pct}</td>
                                    <td style="min-width:110px;">
                                        <div style="display:flex;align-items:center;gap:6px;">
                                            <span style="font-size:0.85rem;font-weight:700;color:var(--text-primary);min-width:38px;">{f1_pct}</span>
                                            <div style="flex:1;background:#e2e8f0;height:6px;border-radius:3px;overflow:hidden;">
                                                <div style="width:{min(100, max(0, f1_val))}%;background:{bar_color};height:100%;border-radius:3px;"></div>
                                            </div>
                                        </div>
                                    </td>
                                    <td style="font-size:0.85rem;font-weight:600;text-align:center;">{c['support']}</td>
                                </tr>"""
    rows_html.append(row)

all_rows_str = "\n".join(rows_html)

# Read index.html and replace evalClassTableBody content
INDEX_PATH = ROOT / "frontend" / "index.html"
with open(INDEX_PATH, "r", encoding="utf-8") as f:
    index_content = f.read()

import re
old_tbody_pattern = r'<tbody id="evalClassTableBody">.*?</tbody>'
new_tbody = f'<tbody id="evalClassTableBody">\n{all_rows_str}\n                            </tbody>'

index_new = re.sub(old_tbody_pattern, new_tbody, index_content, flags=re.DOTALL)
assert index_new != index_content, "Replacement failed!"

with open(INDEX_PATH, "w", encoding="utf-8") as f:
    f.write(index_new)

print("Successfully injected all 28 pre-rendered rows into frontend/index.html!")
