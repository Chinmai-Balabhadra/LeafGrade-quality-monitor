"""
Build robust frontend/js/model_evaluation.js with embedded empirical evaluation data
and dynamic live API synchronization.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "reports" / "model_evaluation_metrics.json"

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    eval_json = json.load(f)

pcm_js = json.dumps(eval_json["per_class_metrics"], indent=4)
om_js = json.dumps(eval_json["overall_metrics"], indent=4)
ds_js = json.dumps(eval_json["dataset_info"], indent=4)

js_content = f"""/**
 * Leaf Quality Classification and Plucking Standard Monitoring System
 * Model Evaluation Dashboard Controller
 * File: frontend/js/model_evaluation.js
 */

const ModelEvaluationController = (function() {{
    // Empirical baseline evaluated on 465 held-out PlantDoc test images
    const FALLBACK_DATASET_INFO = {ds_js};
    const FALLBACK_OVERALL_METRICS = {om_js};
    const FALLBACK_CLASS_METRICS = {pcm_js};

    let evaluationData = null;
    let allClassMetrics = [...FALLBACK_CLASS_METRICS];

    // DOM Elements - Metrics
    let evalAccuracy, evalLoss, evalMacroF1, evalWeightedF1, evalTestSamples, evalClasses;
    let evalClassSearchInput, evalClassTableBody;

    // DOM Elements - Confusion Matrix
    let evalConfusionMatrixImg, btnConfusionMatrixFull, btnZoomMatrixModal;
    let matrixLoadingSpinner, matrixErrorContainer, matrixErrorText, btnRetryMatrixLoad;
    let matrixImageWrapper;
    let confusionMatrixModal, btnCloseMatrixModal, modalConfusionMatrixImg, modalRawPngLink;

    function init() {{
        console.log('[ModelEvaluationController] Initializing Model Performance & Evaluation Dashboard...');

        // Cache DOM elements - Metrics
        evalAccuracy = document.getElementById('evalAccuracy');
        evalLoss = document.getElementById('evalLoss');
        evalMacroF1 = document.getElementById('evalMacroF1');
        evalWeightedF1 = document.getElementById('evalWeightedF1');
        evalTestSamples = document.getElementById('evalTestSamples');
        evalClasses = document.getElementById('evalClasses');

        evalClassSearchInput = document.getElementById('evalClassSearchInput');
        evalClassTableBody = document.getElementById('evalClassTableBody');

        // Cache DOM elements - Confusion Matrix
        evalConfusionMatrixImg = document.getElementById('evalConfusionMatrixImg');
        btnConfusionMatrixFull = document.getElementById('btnConfusionMatrixFull');
        btnZoomMatrixModal = document.getElementById('btnZoomMatrixModal');
        matrixLoadingSpinner = document.getElementById('matrixLoadingSpinner');
        matrixErrorContainer = document.getElementById('matrixErrorContainer');
        matrixErrorText = document.getElementById('matrixErrorText');
        btnRetryMatrixLoad = document.getElementById('btnRetryMatrixLoad');
        matrixImageWrapper = document.getElementById('matrixImageWrapper');

        confusionMatrixModal = document.getElementById('confusionMatrixModal');
        btnCloseMatrixModal = document.getElementById('btnCloseMatrixModal');
        modalConfusionMatrixImg = document.getElementById('modalConfusionMatrixImg');
        modalRawPngLink = document.getElementById('modalRawPngLink');

        // Initial render: Ensure all 28 rows are visible immediately
        renderOverallMetrics(FALLBACK_OVERALL_METRICS, FALLBACK_DATASET_INFO);
        if (evalClassTableBody && (!evalClassTableBody.children || evalClassTableBody.children.length === 0)) {{
            renderClassTable(allClassMetrics);
        }}

        // Bind event listeners
        if (evalClassSearchInput) {{
            evalClassSearchInput.addEventListener('input', handleClassSearch);
        }}

        if (btnRetryMatrixLoad) {{
            btnRetryMatrixLoad.addEventListener('click', loadConfusionMatrix);
        }}

        if (btnZoomMatrixModal) {{
            btnZoomMatrixModal.addEventListener('click', openMatrixModal);
        }}

        if (matrixImageWrapper) {{
            matrixImageWrapper.addEventListener('click', openMatrixModal);
        }}

        if (btnCloseMatrixModal) {{
            btnCloseMatrixModal.addEventListener('click', closeMatrixModal);
        }}

        if (confusionMatrixModal) {{
            confusionMatrixModal.addEventListener('click', (e) => {{
                if (e.target === confusionMatrixModal) {{
                    closeMatrixModal();
                }}
            }});
        }}

        // Close on ESC
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape' && confusionMatrixModal && confusionMatrixModal.classList.contains('active')) {{
                closeMatrixModal();
            }}
        }});

        // Asynchronously synchronize with live backend
        loadModelEvaluation();
        loadConfusionMatrix();
    }}

    async function loadModelEvaluation() {{
        try {{
            let data = null;
            if (typeof ApiService !== 'undefined' && typeof ApiService.getModelEvaluation === 'function') {{
                data = await ApiService.getModelEvaluation();
            }} else {{
                const fallbackUrl = (typeof ApiService !== 'undefined' && ApiService.BASE_URL)
                    ? `${{ApiService.BASE_URL}}/api/model/evaluation`
                    : '/api/model/evaluation';
                const resp = await fetch(fallbackUrl, {{ headers: {{ 'Accept': 'application/json' }} }});
                if (resp.ok) data = await resp.json();
            }}

            if (data && data.status === 'SUCCESS' && Array.isArray(data.per_class_metrics) && data.per_class_metrics.length > 0) {{
                evaluationData = data;
                allClassMetrics = data.per_class_metrics;
                renderOverallMetrics(data.overall_metrics, data.dataset_info);
                renderClassTable(allClassMetrics);
                console.log(`[ModelEvaluationController] Live metrics loaded: ${{allClassMetrics.length}} classes.`);
                return;
            }}
        }} catch (error) {{
            console.warn('[ModelEvaluationController] Live evaluation metrics fetch failed, using empirical baseline:', error.message);
        }}

        // Fallback: Ensure empirical data is rendered
        renderOverallMetrics(FALLBACK_OVERALL_METRICS, FALLBACK_DATASET_INFO);
        renderClassTable(allClassMetrics);
    }}

    function renderOverallMetrics(overall, dataset) {{
        if (!overall) return;

        if (evalAccuracy) {{
            evalAccuracy.textContent = `${{overall.accuracy_percent !== undefined ? overall.accuracy_percent.toFixed(2) : '56.56'}}%`;
        }}
        if (evalLoss) {{
            evalLoss.textContent = overall.loss !== undefined ? overall.loss.toFixed(4) : '1.5560';
        }}
        if (evalMacroF1) {{
            evalMacroF1.textContent = `${{overall.macro_f1_percent !== undefined ? overall.macro_f1_percent.toFixed(2) : '49.74'}}%`;
        }}
        if (evalWeightedF1) {{
            evalWeightedF1.textContent = `${{overall.weighted_f1_percent !== undefined ? overall.weighted_f1_percent.toFixed(2) : '53.80'}}%`;
        }}
        if (evalTestSamples) {{
            evalTestSamples.textContent = overall.total_test_samples !== undefined ? overall.total_test_samples.toString() : '465';
        }}
        if (evalClasses) {{
            evalClasses.textContent = (dataset && (dataset.classes_supported || dataset.total_classes)) ? (dataset.classes_supported || dataset.total_classes).toString() : '28';
        }}
    }}

    function renderClassTable(classes) {{
        if (!evalClassTableBody) return;

        if (!classes || classes.length === 0) {{
            evalClassTableBody.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center py-4 text-muted">
                        <i class="fas fa-search" style="margin-right:6px;"></i> No class evaluation metrics match your search query.
                    </td>
                </tr>
            `;
            return;
        }}

        let html = '';
        classes.forEach(c => {{
            const vh = c.visual_health || 'Diseased';
            const isHealthy = vh === 'Healthy';
            let healthBadge = '';

            if (isHealthy) {{
                healthBadge = '<span class="status-badge badge-healthy" style="padding:2px 8px;font-size:0.75rem;"><i class="fas fa-heart"></i> Healthy</span>';
            }} else if (vh === 'Diseased') {{
                healthBadge = '<span class="status-badge badge-diseased" style="padding:2px 8px;font-size:0.75rem;"><i class="fas fa-virus"></i> Diseased</span>';
            }} else {{
                healthBadge = `<span class="status-badge" style="background:rgba(217,119,6,0.1);color:#d97706;border:1px solid rgba(217,119,6,0.3);padding:2px 8px;font-size:0.75rem;"><i class="fas fa-bug"></i> ${{escapeHtml(vh)}}</span>`;
            }}

            const precVal = typeof c.precision === 'number' ? c.precision : 0;
            const recVal = typeof c.recall === 'number' ? c.recall : 0;
            const f1Val = typeof c.f1_score === 'number' ? c.f1_score : 0;
            const supVal = typeof c.support === 'number' ? c.support : 0;

            const precStr = (precVal * 100).toFixed(1) + '%';
            const recStr = (recVal * 100).toFixed(1) + '%';
            const f1Pct = (f1Val * 100).toFixed(1);

            let barColor = '#16a34a';
            if (f1Val < 0.40) {{
                barColor = '#dc2626';
            }} else if (f1Val < 0.60) {{
                barColor = '#d97706';
            }}

            const cleanName = escapeHtml((c.class_name || '').replace(/_/g, ' '));
            const cleanCond = escapeHtml(c.condition || cleanName);
            const speciesVal = escapeHtml(c.species || '');

            html += `
                <tr data-class-name="${{cleanName.toLowerCase()}}" data-species="${{speciesVal.toLowerCase()}}" data-health="${{vh.toLowerCase()}}">
                    <td style="font-weight:600;font-size:0.85rem;color:var(--text-primary);">
                        <div>${{cleanName}}</div>
                        <div style="font-size:0.75rem;color:var(--text-muted);font-weight:400;">${{cleanCond}}</div>
                    </td>
                    <td>${{healthBadge}}</td>
                    <td style="font-size:0.85rem;font-weight:600;">${{precStr}}</td>
                    <td style="font-size:0.85rem;font-weight:600;">${{recStr}}</td>
                    <td style="min-width:110px;">
                        <div style="display:flex;align-items:center;gap:6px;">
                            <span style="font-size:0.85rem;font-weight:700;color:var(--text-primary);min-width:38px;">${{f1Pct}}%</span>
                            <div style="flex:1;background:#e2e8f0;height:6px;border-radius:3px;overflow:hidden;">
                                <div style="width:${{Math.min(100, Math.max(0, parseFloat(f1Pct)))}}%;background:${{barColor}};height:100%;border-radius:3px;"></div>
                            </div>
                        </div>
                    </td>
                    <td style="font-size:0.85rem;font-weight:600;text-align:center;">${{supVal}}</td>
                </tr>
            `;
        }});

        evalClassTableBody.innerHTML = html;
    }}

    function handleClassSearch() {{
        if (!evalClassSearchInput) return;
        const query = evalClassSearchInput.value.toLowerCase().trim();

        if (!query) {{
            renderClassTable(allClassMetrics);
            return;
        }}

        const filtered = allClassMetrics.filter(c => {{
            const nameMatch = c.class_name && c.class_name.toLowerCase().includes(query);
            const speciesMatch = c.species && c.species.toLowerCase().includes(query);
            const conditionMatch = c.condition && c.condition.toLowerCase().includes(query);
            const healthMatch = c.visual_health && c.visual_health.toLowerCase().includes(query);
            return nameMatch || speciesMatch || conditionMatch || healthMatch;
        }});

        renderClassTable(filtered);
    }}

    /**
     * Loads the 28x28 confusion matrix image with loading states and fallback recovery.
     */
    function loadConfusionMatrix() {{
        const matrixUrl = (typeof ApiService !== 'undefined' && typeof ApiService.getConfusionMatrixUrl === 'function')
            ? ApiService.getConfusionMatrixUrl()
            : '/api/model/confusion-matrix';
        const fallbackUrl = (typeof ApiService !== 'undefined' && ApiService.BASE_URL)
            ? `${{ApiService.BASE_URL}}/assets/confusion_matrix.png`
            : 'assets/confusion_matrix.png';

        console.log('[ModelEvaluationController] Requesting Confusion Matrix from:', matrixUrl);

        if (btnConfusionMatrixFull) {{
            btnConfusionMatrixFull.href = matrixUrl;
        }}
        if (modalRawPngLink) {{
            modalRawPngLink.href = matrixUrl;
        }}

        if (!evalConfusionMatrixImg) return;

        // Show spinner, hide error and image
        if (matrixLoadingSpinner) matrixLoadingSpinner.style.display = 'flex';
        if (matrixErrorContainer) matrixErrorContainer.style.display = 'none';
        if (matrixImageWrapper) matrixImageWrapper.style.display = 'none';

        const testImg = new Image();
        testImg.onload = function() {{
            evalConfusionMatrixImg.src = matrixUrl;
            if (modalConfusionMatrixImg) modalConfusionMatrixImg.src = matrixUrl;
            if (matrixLoadingSpinner) matrixLoadingSpinner.style.display = 'none';
            if (matrixErrorContainer) matrixErrorContainer.style.display = 'none';
            if (matrixImageWrapper) matrixImageWrapper.style.display = 'block';
            console.log('[ModelEvaluationController] Confusion Matrix successfully loaded.');
        }};

        testImg.onerror = function() {{
            console.warn('[ModelEvaluationController] Primary endpoint failed, attempting fallback to static asset:', fallbackUrl);
            const fallbackImg = new Image();
            fallbackImg.onload = function() {{
                evalConfusionMatrixImg.src = fallbackUrl;
                if (modalConfusionMatrixImg) modalConfusionMatrixImg.src = fallbackUrl;
                if (btnConfusionMatrixFull) btnConfusionMatrixFull.href = fallbackUrl;
                if (modalRawPngLink) modalRawPngLink.href = fallbackUrl;
                if (matrixLoadingSpinner) matrixLoadingSpinner.style.display = 'none';
                if (matrixErrorContainer) matrixErrorContainer.style.display = 'none';
                if (matrixImageWrapper) matrixImageWrapper.style.display = 'block';
                console.log('[ModelEvaluationController] Confusion Matrix loaded via fallback asset route.');
            }};

            fallbackImg.onerror = function(err) {{
                console.error('[ModelEvaluationController] Failed to load confusion matrix from both endpoints:', err);
                if (matrixLoadingSpinner) matrixLoadingSpinner.style.display = 'none';
                if (matrixImageWrapper) matrixImageWrapper.style.display = 'none';
                if (matrixErrorContainer) {{
                    matrixErrorContainer.style.display = 'block';
                    if (matrixErrorText) {{
                        matrixErrorText.textContent = `Could not reach ${{matrixUrl}}. Ensure the Flask server is running at http://127.0.0.1:5000.`;
                    }}
                }}
            }};
            fallbackImg.src = fallbackUrl;
        }};

        testImg.src = matrixUrl;
    }}

    function openMatrixModal() {{
        if (!confusionMatrixModal) return;
        const currentSrc = evalConfusionMatrixImg && evalConfusionMatrixImg.src
            ? evalConfusionMatrixImg.src
            : ((typeof ApiService !== 'undefined' && typeof ApiService.getConfusionMatrixUrl === 'function')
                ? ApiService.getConfusionMatrixUrl()
                : '/api/model/confusion-matrix');

        if (modalConfusionMatrixImg) {{
            modalConfusionMatrixImg.src = currentSrc;
        }}
        confusionMatrixModal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }}

    function closeMatrixModal() {{
        if (!confusionMatrixModal) return;
        confusionMatrixModal.classList.remove('active');
        document.body.style.overflow = '';
    }}

    function escapeHtml(str) {{
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }}

    return {{
        init,
        loadModelEvaluation,
        loadConfusionMatrix,
        openMatrixModal,
        closeMatrixModal
    }};
}})();

// Explicitly attach to window
window.ModelEvaluationController = ModelEvaluationController;
"""

JS_PATH = ROOT / "frontend" / "js" / "model_evaluation.js"
with open(JS_PATH, "w", encoding="utf-8") as f:
    f.write(js_content)

print(f"Successfully wrote {len(js_content)} bytes to {JS_PATH}")
