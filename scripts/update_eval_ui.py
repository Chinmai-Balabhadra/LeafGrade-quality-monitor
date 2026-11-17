"""
Script to update frontend/index.html, frontend/js/model_evaluation.js, and frontend/css/responsive.css
with enhanced Per-Class Evaluation Metrics Table width and large Confusion Matrix preview.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def build_row_html(c):
    class_name = c["class_name"].replace("_", " ")
    condition = c.get("condition", class_name)
    species = c.get("species", "")
    visual_health = c.get("visual_health", "Diseased")
    precision_pct = f"{c.get('precision', 0) * 100:.1f}%"
    recall_pct = f"{c.get('recall', 0) * 100:.1f}%"
    f1_val = c.get("f1_score", 0)
    f1_pct = f"{f1_val * 100:.1f}"
    support = c.get("support", 0)

    if visual_health == "Healthy":
        badge = '<span class="status-badge badge-healthy" style="padding:3px 10px;font-size:0.75rem;font-weight:600;"><i class="fas fa-heart"></i> Healthy</span>'
    elif visual_health == "Diseased":
        badge = '<span class="status-badge badge-diseased" style="padding:3px 10px;font-size:0.75rem;font-weight:600;"><i class="fas fa-virus"></i> Diseased</span>'
    else:
        badge = f'<span class="status-badge" style="background:rgba(217,119,6,0.1);color:#d97706;border:1px solid rgba(217,119,6,0.3);padding:3px 10px;font-size:0.75rem;font-weight:600;"><i class="fas fa-bug"></i> {visual_health}</span>'

    if f1_val < 0.40:
        bar_color = "#dc2626"
    elif f1_val < 0.60:
        bar_color = "#d97706"
    else:
        bar_color = "#16a34a"

    f1_width = min(100.0, max(0.0, float(f1_pct)))

    return f'''                                <tr data-class-name="{class_name.lower()}" data-species="{species.lower()}" data-health="{visual_health.lower()}">
                                    <td class="eval-col-name">
                                        <span style="font-weight:700;color:var(--text-primary);font-size:0.88rem;">{class_name}</span>
                                        <span style="font-size:0.78rem;color:var(--text-muted);font-weight:400;margin-left:6px;">&bull; {condition}</span>
                                    </td>
                                    <td class="eval-col-health">{badge}</td>
                                    <td class="eval-col-stat" style="font-size:0.88rem;font-weight:600;">{precision_pct}</td>
                                    <td class="eval-col-stat" style="font-size:0.88rem;font-weight:600;">{recall_pct}</td>
                                    <td class="eval-col-f1">
                                        <div style="display:flex;align-items:center;gap:8px;">
                                            <span style="font-size:0.85rem;font-weight:700;color:var(--text-primary);min-width:42px;">{f1_pct}%</span>
                                            <div style="flex:1;background:#e2e8f0;height:6px;border-radius:3px;overflow:hidden;min-width:60px;">
                                                <div style="width:{f1_width}%;background:{bar_color};height:100%;border-radius:3px;"></div>
                                            </div>
                                        </div>
                                    </td>
                                    <td style="font-size:0.88rem;font-weight:700;text-align:center;">{support}</td>
                                </tr>'''

def main():
    with open(ROOT / "reports" / "model_evaluation_metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    classes = data["per_class_metrics"]
    assert len(classes) == 28, f"Expected 28 classes, got {len(classes)}"
    total_sup = sum(c["support"] for c in classes)
    assert total_sup == 465, f"Expected 465 support, got {total_sup}"

    rows_html = "\n".join(build_row_html(c) for c in classes)

    new_section_html = f'''            <!-- Per-Class Classification Report Table (Full Width) -->
            <div class="eval-card-full">
                <div class="history-toolbar" style="border-bottom:1px solid var(--border-color);display:flex;justify-content:space-between;align-items:center;padding:1.15rem 1.5rem;flex-wrap:wrap;gap:1rem;">
                    <div style="display:flex;align-items:center;gap:0.75rem;">
                        <div style="width:36px;height:36px;border-radius:8px;background:rgba(37,99,235,0.1);display:flex;align-items:center;justify-content:center;color:#2563eb;font-size:1.1rem;">
                            <i class="fas fa-table"></i>
                        </div>
                        <div>
                            <h4 style="margin:0;font-size:1.1rem;font-weight:700;color:var(--text-primary);">
                                Per-Class Evaluation Metrics (28 Classes)
                            </h4>
                            <p style="margin:2px 0 0;font-size:0.8rem;color:var(--text-muted);">
                                Held-out test set performance breakdown (N=465 images)
                            </p>
                        </div>
                    </div>
                    <div class="search-box-wrap" style="width:260px;">
                        <i class="fas fa-search search-icon"></i>
                        <input type="text" id="evalClassSearchInput" class="form-input search-input" placeholder="Search class, species, or condition..." style="font-size:0.85rem;padding:0.45rem 0.75rem 0.45rem 2.2rem;border-radius:var(--radius-md);">
                    </div>
                </div>
                <div class="eval-table-container">
                    <table class="eval-table history-table" id="evalClassTable">
                        <thead style="position: sticky; top: 0; background: #f8fafc; z-index: 2; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                            <tr>
                                <th class="eval-col-name">Class Name &amp; Biological Condition</th>
                                <th class="eval-col-health">Visual Health</th>
                                <th class="eval-col-stat">Precision</th>
                                <th class="eval-col-stat">Recall</th>
                                <th class="eval-col-f1">F1-Score</th>
                                <th class="eval-col-support" style="text-align:center;">Support</th>
                            </tr>
                        </thead>
                        <tbody id="evalClassTableBody">
{rows_html}
                        </tbody>
                    </table>
                </div>
                <div style="padding:0.75rem 1.5rem;background:#f8fafc;border-top:1px solid var(--border-color);display:flex;justify-content:space-between;align-items:center;font-size:0.82rem;color:var(--text-muted);flex-wrap:wrap;gap:0.5rem;">
                    <span>Showing all 28 classes evaluated on PlantDoc test partition</span>
                    <span>Total test support: <strong>465 samples</strong> &bull; Bilinear interpolation</span>
                </div>
            </div>

            <!-- 28x28 Confusion Matrix Large Preview Card (Full Width) -->
            <div class="matrix-preview-card">
                <div class="history-toolbar" style="border-bottom:1px solid var(--border-color);display:flex;justify-content:space-between;align-items:center;padding:1.15rem 1.5rem;flex-wrap:wrap;gap:1rem;">
                    <div style="display:flex;align-items:center;gap:0.75rem;">
                        <div style="width:36px;height:36px;border-radius:8px;background:rgba(37,99,235,0.1);display:flex;align-items:center;justify-content:center;color:#2563eb;font-size:1.1rem;">
                            <i class="fas fa-th"></i>
                        </div>
                        <div>
                            <h4 style="margin:0;font-size:1.1rem;font-weight:700;color:var(--text-primary);">
                                28&times;28 Confusion Matrix Heatmap
                            </h4>
                            <p style="margin:2px 0 0;font-size:0.8rem;color:var(--text-muted);">
                                MobileNetV2 classification predictions across all 28 classes (N=465)
                            </p>
                        </div>
                    </div>
                    <div style="display:flex;gap:0.6rem;align-items:center;">
                        <button type="button" id="btnZoomMatrixModal" class="btn btn-outline" style="font-size:0.85rem;padding:0.45rem 0.9rem;" title="Enlarge in lightbox modal">
                            <i class="fas fa-search-plus"></i> Inspect
                        </button>
                        <a href="/api/model/confusion-matrix" target="_blank" id="btnConfusionMatrixFull" class="btn btn-secondary" style="font-size:0.85rem;padding:0.45rem 1rem;" title="Open raw PNG in new tab">
                            <i class="fas fa-external-link-alt"></i> Full Resolution
                        </a>
                    </div>
                </div>
                <div style="padding: 1.5rem; text-align: center; position: relative;">
                    <!-- Loading spinner -->
                    <div id="matrixLoadingSpinner" style="display:none;padding:3.5rem 1rem;flex-direction:column;align-items:center;justify-content:center;">
                        <div class="loading-spinner"></div>
                        <p style="margin-top:0.85rem;font-size:0.9rem;color:var(--text-muted);">Loading 28&times;28 Confusion Matrix Heatmap...</p>
                    </div>

                    <!-- Error container -->
                    <div id="matrixErrorContainer" style="display:none;padding:3rem 1rem;text-align:center;">
                        <i class="fas fa-exclamation-triangle" style="font-size:2.5rem;color:var(--warning);margin-bottom:0.75rem;"></i>
                        <h5 style="margin:0 0 0.5rem;font-size:1rem;font-weight:700;color:var(--text-primary);">Matrix Image Unavailable</h5>
                        <p id="matrixErrorText" style="font-size:0.85rem;color:var(--text-muted);margin:0 0 1rem;max-width:360px;margin-left:auto;margin-right:auto;">
                            The confusion matrix could not be retrieved from the server.
                        </p>
                        <button type="button" id="btnRetryMatrixLoad" class="btn btn-sm btn-primary">
                            <i class="fas fa-redo"></i> Retry Loading
                        </button>
                    </div>

                    <!-- Matrix Image Preview with zoom trigger -->
                    <div class="matrix-canvas-wrapper" id="matrixImageWrapper" title="Click to enlarge matrix in modal">
                        <img src="/api/model/confusion-matrix" alt="MobileNetV2 28x28 Confusion Matrix" id="evalConfusionMatrixImg">
                        <div style="position:absolute;bottom:12px;right:12px;background:rgba(15,23,42,0.8);color:#ffffff;font-size:0.8rem;padding:5px 10px;border-radius:6px;pointer-events:none;display:flex;align-items:center;gap:6px;backdrop-filter:blur(4px);">
                            <i class="fas fa-search-plus"></i> Click to Enlarge / Inspect
                        </div>
                    </div>

                    <p style="margin:1rem 0 0;font-size:0.85rem;color:var(--text-muted);max-width:720px;margin-left:auto;margin-right:auto;line-height:1.5;">
                        <strong>Rows:</strong> Ground-truth test annotations &bull; <strong>Columns:</strong> Model predictions. Diagonal cells represent correct classifications. Evaluated with consistent bilinear interpolation matching model training.
                    </p>
                </div>
            </div>'''

    index_path = ROOT / "frontend" / "index.html"
    index_content = index_path.read_text(encoding="utf-8")

    # Regex find the block: from table card comment to '</section>'
    old_grid_pattern = re.compile(
        r'([ \t]*<!-- (?:Performance Table & Confusion Matrix Card Grid|Per-Class Classification Report Table \(Full Width\)) -->.*?)(?=\n[ \t]*</div>\s*\n[ \t]*</section>)',
        re.DOTALL
    )

    m = old_grid_pattern.search(index_content)
    if not m:
        print("ERROR: Could not locate grid block in frontend/index.html")
        return

    updated_index = index_content[:m.start()] + new_section_html + index_content[m.end():]
    index_path.write_text(updated_index, encoding="utf-8")
    print("SUCCESS: Updated frontend/index.html with full-width table and matrix preview cards.")

if __name__ == "__main__":
    main()
