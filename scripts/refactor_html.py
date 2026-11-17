"""
Cleanly rebuild frontend/index.html to remove duplicate History sections
and insert the AI Model Performance & Evaluation Dashboard.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
INDEX_HTML = ROOT / "frontend" / "index.html"

with open(INDEX_HTML, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Locate line ranges for:
# 1. Start through end of #analysis
# 2. First #history section (from <section class="section" id="history" to closing </section>)
# 3. First #historyDetailModal (from <div id="historyDetailModal" to closing </div></div></div>)
# 4. #features (from <section class="section" id="features" to </section>)
# 5. #how-it-works (from <section class="section" id="how-it-works" to </section>)
# 6. #about (from <section class="section" id="about" to </section>)
# 7. Footer (from <footer class="footer" to </html>)

content = "".join(lines)

# Find analysis end
idx_analysis_end = content.find('</section>', content.find('id="analysis"')) + len('</section>')
pre_history = content[:idx_analysis_end]

# Find history section
idx_hist_start = content.find('<section class="section" id="history"')
idx_hist_end = content.find('</section>', idx_hist_start) + len('</section>')
history_section = content[idx_hist_start:idx_hist_end]

# Find historyDetailModal
idx_modal_start = content.find('<div id="historyDetailModal"', idx_hist_end)
# Find the closing for this modal
# The modal structure is <div id="historyDetailModal" ...> <div class="modal-dialog"> ... <div class="modal-footer">...</div> </div> </div>
idx_modal_dialog = content.find('<div class="modal-dialog">', idx_modal_start)
idx_modal_footer = content.find('<div class="modal-footer">', idx_modal_dialog)
idx_modal_footer_end = content.find('</div>', idx_modal_footer) + len('</div>')
idx_modal_dialog_end = content.find('</div>', idx_modal_footer_end) + len('</div>')
idx_modal_end = content.find('</div>', idx_modal_dialog_end) + len('</div>')
modal_section = content[idx_modal_start:idx_modal_end]

# Find features section
idx_feat_start = content.find('<section class="section" id="features"')
idx_feat_end = content.find('</section>', idx_feat_start) + len('</section>')
features_section = content[idx_feat_start:idx_feat_end]

# Find how-it-works section
idx_hiw_start = content.find('<section class="section" id="how-it-works"')
idx_hiw_end = content.find('</section>', idx_hiw_start) + len('</section>')
hiw_section = content[idx_hiw_start:idx_hiw_end]

# Find about section
idx_about_start = content.find('<section class="section" id="about"')
idx_about_end = content.find('</section>', idx_about_start) + len('</section>')
about_section = content[idx_about_start:idx_about_end]

# Find footer through end of document
idx_footer_start = content.find('<footer class="footer"')
footer_section = content[idx_footer_start:]

# Ensure Model Performance is in footer links
if '<li><a href="#model-performance">Model Performance</a></li>' not in footer_section:
    footer_section = footer_section.replace(
        '<li><a href="#history">Inspection History</a></li>',
        '<li><a href="#history">Inspection History</a></li>\n                        <li><a href="#model-performance">Model Performance</a></li>'
    )

# Ensure model_evaluation.js script is included
if '<script src="js/model_evaluation.js"></script>' not in footer_section:
    footer_section = footer_section.replace(
        '<script src="js/main.js"></script>',
        '<script src="js/model_evaluation.js"></script>\n    <script src="js/main.js"></script>'
    )

MODEL_PERFORMANCE_HTML = """
    <!-- ====================================================================
         AI Model Performance & Evaluation Dashboard
         ==================================================================== -->
    <section class="section" id="model-performance" style="background:#ffffff;border-top:1px solid var(--border-color);">
        <div class="container">
            <div class="section-header">
                <div class="section-subtitle">Empirical Validation &amp; Metrics</div>
                <h2 class="section-title">AI Model Performance &amp; Evaluation</h2>
                <p class="section-desc">Quantitative evaluation results of the MobileNetV2 architecture on 465 held-out PlantDoc test samples using bilinear interpolation preprocessing.</p>
            </div>

            <!-- Metric Cards Grid -->
            <div class="history-stats-grid" style="grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); margin-bottom: 2rem;">
                <div class="stat-card">
                    <div class="stat-icon-wrap" style="color:#16a34a;background:rgba(22,163,74,0.1);"><i class="fas fa-bullseye"></i></div>
                    <div class="stat-content">
                        <span class="stat-label">Test Accuracy</span>
                        <h3 class="stat-value" id="evalAccuracy">56.56%</h3>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-icon-wrap" style="color:#d97706;background:rgba(217,119,6,0.1);"><i class="fas fa-calculator"></i></div>
                    <div class="stat-content">
                        <span class="stat-label">Loss (CCE)</span>
                        <h3 class="stat-value" id="evalLoss">1.5560</h3>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-icon-wrap" style="color:#2563eb;background:rgba(37,99,235,0.1);"><i class="fas fa-balance-scale"></i></div>
                    <div class="stat-content">
                        <span class="stat-label">Macro F1</span>
                        <h3 class="stat-value" id="evalMacroF1">49.74%</h3>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-icon-wrap" style="color:#7c3aed;background:rgba(124,58,237,0.1);"><i class="fas fa-layer-group"></i></div>
                    <div class="stat-content">
                        <span class="stat-label">Weighted F1</span>
                        <h3 class="stat-value" id="evalWeightedF1">53.80%</h3>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-icon-wrap" style="color:#0891b2;background:rgba(8,145,178,0.1);"><i class="fas fa-images"></i></div>
                    <div class="stat-content">
                        <span class="stat-label">Test Samples</span>
                        <h3 class="stat-value" id="evalTestSamples">465</h3>
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-icon-wrap" style="color:#059669;background:rgba(5,150,105,0.1);"><i class="fas fa-th-list"></i></div>
                    <div class="stat-content">
                        <span class="stat-label">Crop Classes</span>
                        <h3 class="stat-value" id="evalClasses">28</h3>
                    </div>
                </div>
            </div>

            <!-- Model Info & Dataset Breakdown Panel -->
            <div style="display:grid;grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; margin-bottom: 2rem;">
                <div class="stat-card" style="display:block;padding:1.5rem;">
                    <h4 style="margin:0 0 1rem;font-size:1.05rem;color:var(--text-primary);display:flex;align-items:center;gap:0.5rem;">
                        <i class="fas fa-brain" style="color:var(--primary);"></i> Model Architecture &amp; Hyperparameters
                    </h4>
                    <ul style="list-style:none;padding:0;margin:0;font-size:0.9rem;color:var(--text-secondary);line-height:1.8;">
                        <li><strong>Base Backbone:</strong> MobileNetV2 (ImageNet Pre-trained)</li>
                        <li><strong>Input Resolution:</strong> 224 &times; 224 &times; 3</li>
                        <li><strong>Interpolation:</strong> Bilinear (<code>cv2.INTER_LINEAR</code>)</li>
                        <li><strong>Rescaling Range:</strong> [-1.0, 1.0] (Normalized)</li>
                        <li><strong>Total Parameters:</strong> 2,293,852</li>
                        <li><strong>Fine-Tuning:</strong> Top 34 layers fine-tuned (LR: 1e-4)</li>
                    </ul>
                </div>

                <div class="stat-card" style="display:block;padding:1.5rem;">
                    <h4 style="margin:0 0 1rem;font-size:1.05rem;color:var(--text-primary);display:flex;align-items:center;gap:0.5rem;">
                        <i class="fas fa-database" style="color:#2563eb;"></i> PlantDoc Dataset Stratification
                    </h4>
                    <ul style="list-style:none;padding:0;margin:0;font-size:0.9rem;color:var(--text-secondary);line-height:1.8;">
                        <li><strong>Total Cleaned Images:</strong> 2,916 images across 28 classes</li>
                        <li><strong>Training Set:</strong> 2,028 images (~70%)</li>
                        <li><strong>Validation Set:</strong> 423 images (~14.5%)</li>
                        <li><strong>Held-out Test Set:</strong> 465 images (~15.5%)</li>
                        <li><strong>Healthy Classes:</strong> 10 / 28 classes</li>
                        <li><strong>Diseased Classes:</strong> 18 / 28 classes</li>
                    </ul>
                </div>
            </div>

            <!-- Performance Table & Confusion Matrix Card Grid -->
            <div style="display:grid;grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 1.5rem; align-items: start;">
                <!-- Left: Per-Class Classification Report Table -->
                <div class="history-card">
                    <div class="history-toolbar" style="border-bottom:1px solid var(--border-color);display:flex;justify-content:space-between;align-items:center;padding:1rem 1.25rem;">
                        <h4 style="margin:0;font-size:1.05rem;font-weight:700;display:flex;align-items:center;gap:0.5rem;">
                            <i class="fas fa-table" style="color:var(--primary);"></i> Per-Class Evaluation Metrics (28 Classes)
                        </h4>
                        <div class="search-box-wrap" style="width:180px;">
                            <i class="fas fa-search search-icon"></i>
                            <input type="text" id="evalClassSearchInput" class="form-input search-input" placeholder="Filter class..." style="font-size:0.85rem;padding:0.35rem 0.6rem 0.35rem 2rem;">
                        </div>
                    </div>
                    <div class="table-responsive" style="max-height: 480px; overflow-y: auto;">
                        <table class="history-table" id="evalClassTable">
                            <thead style="position: sticky; top: 0; background: #f8fafc; z-index: 1;">
                                <tr>
                                    <th>Class Name</th>
                                    <th>Health</th>
                                    <th>Precision</th>
                                    <th>Recall</th>
                                    <th>F1-Score</th>
                                    <th>Support</th>
                                </tr>
                            </thead>
                            <tbody id="evalClassTableBody">
                                <!-- Populated dynamically by ModelEvaluationController -->
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Right: Confusion Matrix Viewer Card -->
                <div class="history-card">
                    <div class="history-toolbar" style="border-bottom:1px solid var(--border-color);display:flex;justify-content:space-between;align-items:center;padding:1rem 1.25rem;">
                        <h4 style="margin:0;font-size:1.05rem;font-weight:700;display:flex;align-items:center;gap:0.5rem;">
                            <i class="fas fa-th" style="color:#2563eb;"></i> 28&times;28 Confusion Matrix
                        </h4>
                        <a href="/api/model/confusion-matrix" target="_blank" class="btn btn-sm btn-secondary" style="font-size:0.8rem;padding:0.35rem 0.75rem;">
                            <i class="fas fa-external-link-alt"></i> Full Resolution
                        </a>
                    </div>
                    <div style="padding: 1.25rem; text-align: center;">
                        <div style="position:relative;background:#f8fafc;border:1px solid var(--border-color);border-radius:var(--radius-md);overflow:hidden;box-shadow:inset 0 1px 3px rgba(0,0,0,0.05);">
                            <img src="/api/model/confusion-matrix" alt="MobileNetV2 28x28 Confusion Matrix" id="evalConfusionMatrixImg" style="width:100%;height:auto;max-height:430px;object-fit:contain;display:block;" onerror="this.src='reports/confusion_matrix.png';">
                        </div>
                        <p style="margin:0.75rem 0 0;font-size:0.8rem;color:var(--text-muted);">
                            Rows: Actual Ground Truth &bull; Columns: MobileNetV2 Model Predictions on Held-Out Test Set (N=465)
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </section>
"""

new_html = f"""{pre_history}

    <!-- ====================================================================
         Historical Analysis & Database Dashboard Section
         ==================================================================== -->
    {history_section}

    <!-- Modal: Record Diagnostic Detail -->
    {modal_section}

{MODEL_PERFORMANCE_HTML}

    <!-- ====================================================================
         Features Section
         ==================================================================== -->
    {features_section}

    <!-- ====================================================================
         How It Works Section
         ==================================================================== -->
    {hiw_section}

    <!-- ====================================================================
         About Section
         ==================================================================== -->
    {about_section}

    <!-- ====================================================================
         Footer
         ==================================================================== -->
    {footer_section}
"""

with open(INDEX_HTML, "w", encoding="utf-8") as f:
    f.write(new_html)

print("HTML refactored successfully!")
