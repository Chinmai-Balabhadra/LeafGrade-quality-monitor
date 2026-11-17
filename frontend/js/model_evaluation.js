/**
 * Leaf Quality Classification and Plucking Standard Monitoring System
 * Model Evaluation Dashboard Controller
 * File: frontend/js/model_evaluation.js
 */

const ModelEvaluationController = (function() {
    // Empirical baseline evaluated on 465 held-out PlantDoc test images
    const FALLBACK_DATASET_INFO = {
    "name": "PlantDoc (Cleaned & Stratified)",
    "total_images": 2916,
    "training_images": 2028,
    "validation_images": 423,
    "test_images": 465,
    "healthy_classes": 10,
    "diseased_classes": 18
};
    const FALLBACK_OVERALL_METRICS = {
    "accuracy": 0.5656,
    "accuracy_percent": 56.56,
    "loss": 1.556,
    "total_test_samples": 465,
    "correct_predictions": 263,
    "incorrect_predictions": 202,
    "macro_precision": 0.5106,
    "macro_precision_percent": 51.06,
    "macro_recall": 0.5233,
    "macro_recall_percent": 52.33,
    "macro_f1": 0.4974,
    "macro_f1_percent": 49.74,
    "weighted_precision": 0.5467,
    "weighted_precision_percent": 54.67,
    "weighted_recall": 0.5656,
    "weighted_recall_percent": 56.56,
    "weighted_f1": 0.538,
    "weighted_f1_percent": 53.8
};
    const FALLBACK_CLASS_METRICS = [
    {
        "class_index": 0,
        "class_name": "Apple_Scab_Leaf",
        "species": "Apple",
        "condition": "Apple scab (Venturia inaequalis)",
        "visual_health": "Diseased",
        "quality_category": "Poor Quality (Visually Diseased/Damaged)",
        "support": 15,
        "true_positives": 8,
        "false_positives": 20,
        "false_negatives": 7,
        "precision": 0.2857,
        "recall": 0.5333,
        "f1_score": 0.3721,
        "precision_pct": 28.57,
        "recall_pct": 53.33,
        "f1_pct": 37.21
    },
    {
        "class_index": 1,
        "class_name": "Apple_leaf",
        "species": "Apple",
        "condition": "Healthy foliage",
        "visual_health": "Healthy",
        "quality_category": "Good Quality (Visually Healthy)",
        "support": 14,
        "true_positives": 11,
        "false_positives": 4,
        "false_negatives": 3,
        "precision": 0.7333,
        "recall": 0.7857,
        "f1_score": 0.7586,
        "precision_pct": 73.33,
        "recall_pct": 78.57,
        "f1_pct": 75.86
    },
    {
        "class_index": 2,
        "class_name": "Apple_rust_leaf",
        "species": "Apple",
        "condition": "Cedar apple rust infection",
        "visual_health": "Diseased",
        "quality_category": "Poor Quality (Visually Diseased/Damaged)",
        "support": 17,
        "true_positives": 16,
        "false_positives": 7,
        "false_negatives": 1,
        "precision": 0.6957,
        "recall": 0.9412,
        "f1_score": 0.8,
        "precision_pct": 69.57,
        "recall_pct": 94.12,
        "f1_pct": 80.0
    },
    {
        "class_index": 3,
        "class_name": "Bell_pepper_leaf",
        "species": "Bell Pepper",
        "condition": "Healthy foliage",
        "visual_health": "Healthy",
        "quality_category": "Good Quality (Visually Healthy)",
        "support": 7,
        "true_positives": 5,
        "false_positives": 3,
        "false_negatives": 2,
        "precision": 0.625,
        "recall": 0.7143,
        "f1_score": 0.6667,
        "precision_pct": 62.5,
        "recall_pct": 71.43,
        "f1_pct": 66.67
    },
    {
        "class_index": 4,
        "class_name": "Bell_pepper_leaf_spot",
        "species": "Bell Pepper",
        "condition": "Bacterial leaf spot (Xanthomonas)",
        "visual_health": "Diseased",
        "quality_category": "Poor Quality (Visually Diseased/Damaged)",
        "support": 13,
        "true_positives": 3,
        "false_positives": 4,
        "false_negatives": 10,
        "precision": 0.4286,
        "recall": 0.2308,
        "f1_score": 0.3,
        "precision_pct": 42.86,
        "recall_pct": 23.08,
        "f1_pct": 30.0
    },
    {
        "class_index": 5,
        "class_name": "Blueberry_leaf",
        "species": "Blueberry",
        "condition": "Healthy foliage",
        "visual_health": "Healthy",
        "quality_category": "Good Quality (Visually Healthy)",
        "support": 18,
        "true_positives": 8,
        "false_positives": 8,
        "false_negatives": 10,
        "precision": 0.5,
        "recall": 0.4444,
        "f1_score": 0.4706,
        "precision_pct": 50.0,
        "recall_pct": 44.44,
        "f1_pct": 47.06
    },
    {
        "class_index": 6,
        "class_name": "Cherry_leaf",
        "species": "Cherry",
        "condition": "Healthy foliage",
        "visual_health": "Healthy",
        "quality_category": "Good Quality (Visually Healthy)",
        "support": 10,
        "true_positives": 6,
        "false_positives": 5,
        "false_negatives": 4,
        "precision": 0.5455,
        "recall": 0.6,
        "f1_score": 0.5714,
        "precision_pct": 54.55,
        "recall_pct": 60.0,
        "f1_pct": 57.14
    },
    {
        "class_index": 7,
        "class_name": "Corn_Gray_leaf_spot",
        "species": "Corn (Maize)",
        "condition": "Cercospora zeae-maydis",
        "visual_health": "Diseased",
        "quality_category": "Poor Quality (Visually Diseased/Damaged)",
        "support": 11,
        "true_positives": 5,
        "false_positives": 2,
        "false_negatives": 6,
        "precision": 0.7143,
        "recall": 0.4545,
        "f1_score": 0.5556,
        "precision_pct": 71.43,
        "recall_pct": 45.45,
        "f1_pct": 55.56
    },
    {
        "class_index": 8,
        "class_name": "Corn_leaf_blight",
        "species": "Corn (Maize)",
        "condition": "Northern corn leaf blight (Exserohilum)",
        "visual_health": "Diseased",
        "quality_category": "Poor Quality (Visually Diseased/Damaged)",
        "support": 30,
        "true_positives": 27,
        "false_positives": 11,
        "false_negatives": 3,
        "precision": 0.7105,
        "recall": 0.9,
        "f1_score": 0.7941,
        "precision_pct": 71.05,
        "recall_pct": 90.0,
        "f1_pct": 79.41
    },
    {
        "class_index": 9,
        "class_name": "Corn_rust_leaf",
        "species": "Corn (Maize)",
        "condition": "Common rust (Puccinia sorghi)",
        "visual_health": "Diseased",
        "quality_category": "Poor Quality (Visually Diseased/Damaged)",
        "support": 19,
        "true_positives": 16,
        "false_positives": 11,
        "false_negatives": 3,
        "precision": 0.5926,
        "recall": 0.8421,
        "f1_score": 0.6957,
        "precision_pct": 59.26,
        "recall_pct": 84.21,
        "f1_pct": 69.57
    },
    {
        "class_index": 10,
        "class_name": "Peach_leaf",
        "species": "Peach",
        "condition": "Healthy foliage",
        "visual_health": "Healthy",
        "quality_category": "Good Quality (Visually Healthy)",
        "support": 18,
        "true_positives": 14,
        "false_positives": 11,
        "false_negatives": 4,
        "precision": 0.56,
        "recall": 0.7778,
        "f1_score": 0.6512,
        "precision_pct": 56.0,
        "recall_pct": 77.78,
        "f1_pct": 65.12
    },
    {
        "class_index": 11,
        "class_name": "Potato_leaf_early_blight",
        "species": "Potato",
        "condition": "Early blight (Alternaria solani)",
        "visual_health": "Diseased",
        "quality_category": "Poor Quality (Visually Diseased/Damaged)",
        "support": 27,
        "true_positives": 4,
        "false_positives": 7,
        "false_negatives": 23,
        "precision": 0.3636,
        "recall": 0.1481,
        "f1_score": 0.2105,
        "precision_pct": 36.36,
        "recall_pct": 14.81,
        "f1_pct": 21.05
    },
    {
        "class_index": 12,
        "class_name": "Potato_leaf_late_blight",
        "species": "Potato",
        "condition": "Late blight (Phytophthora infestans)",
        "visual_health": "Diseased",
        "quality_category": "Poor Quality (Visually Diseased/Damaged)",
        "support": 32,
        "true_positives": 14,
        "false_positives": 11,
        "false_negatives": 18,
        "precision": 0.56,
        "recall": 0.4375,
        "f1_score": 0.4912,
        "precision_pct": 56.0,
        "recall_pct": 43.75,
        "f1_pct": 49.12
    },
    {
        "class_index": 13,
        "class_name": "Raspberry_leaf",
        "species": "Raspberry",
        "condition": "Healthy foliage",
        "visual_health": "Healthy",
        "quality_category": "Good Quality (Visually Healthy)",
        "support": 19,
        "true_positives": 11,
        "false_positives": 3,
        "false_negatives": 8,
        "precision": 0.7857,
        "recall": 0.5789,
        "f1_score": 0.6667,
        "precision_pct": 78.57,
        "recall_pct": 57.89,
        "f1_pct": 66.67
    },
    {
        "class_index": 14,
        "class_name": "Soyabean_leaf",
        "species": "Soybean",
        "condition": "Healthy foliage",
        "visual_health": "Healthy",
        "quality_category": "Good Quality (Visually Healthy)",
        "support": 11,
        "true_positives": 4,
        "false_positives": 8,
        "false_negatives": 7,
        "precision": 0.3333,
        "recall": 0.3636,
        "f1_score": 0.3478,
        "precision_pct": 33.33,
        "recall_pct": 36.36,
        "f1_pct": 34.78
    },
    {
        "class_index": 15,
        "class_name": "Squash_Powdery_mildew_leaf",
        "species": "Squash",
        "condition": "Powdery mildew (Podosphaera)",
        "visual_health": "Diseased",
        "quality_category": "Poor Quality (Visually Diseased/Damaged)",
        "support": 20,
        "true_positives": 18,
        "false_positives": 11,
        "false_negatives": 2,
        "precision": 0.6207,
        "recall": 0.9,
        "f1_score": 0.7347,
        "precision_pct": 62.07,
        "recall_pct": 90.0,
        "f1_pct": 73.47
    },
    {
        "class_index": 16,
        "class_name": "Strawberry_leaf",
        "species": "Strawberry",
        "condition": "Healthy foliage",
        "visual_health": "Healthy",
        "quality_category": "Good Quality (Visually Healthy)",
        "support": 15,
        "true_positives": 11,
        "false_positives": 2,
        "false_negatives": 4,
        "precision": 0.8462,
        "recall": 0.7333,
        "f1_score": 0.7857,
        "precision_pct": 84.62,
        "recall_pct": 73.33,
        "f1_pct": 78.57
    },
    {
        "class_index": 17,
        "class_name": "Tomato_Early_blight_leaf",
        "species": "Tomato",
        "condition": "Early blight (Alternaria solani)",
        "visual_health": "Diseased",
        "quality_category": "Poor Quality (Visually Diseased/Damaged)",
        "support": 14,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 14,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "precision_pct": 0.0,
        "recall_pct": 0.0,
        "f1_pct": 0.0
    },
    {
        "class_index": 18,
        "class_name": "Tomato_Septoria_leaf_spot",
        "species": "Tomato",
        "condition": "Septoria leaf spot (Septoria lycopersici)",
        "visual_health": "Diseased",
        "quality_category": "Poor Quality (Visually Diseased/Damaged)",
        "support": 25,
        "true_positives": 15,
        "false_positives": 12,
        "false_negatives": 10,
        "precision": 0.5556,
        "recall": 0.6,
        "f1_score": 0.5769,
        "precision_pct": 55.56,
        "recall_pct": 60.0,
        "f1_pct": 57.69
    },
    {
        "class_index": 19,
        "class_name": "Tomato_leaf",
        "species": "Tomato",
        "condition": "Healthy foliage",
        "visual_health": "Healthy",
        "quality_category": "Good Quality (Visually Healthy)",
        "support": 9,
        "true_positives": 2,
        "false_positives": 1,
        "false_negatives": 7,
        "precision": 0.6667,
        "recall": 0.2222,
        "f1_score": 0.3333,
        "precision_pct": 66.67,
        "recall_pct": 22.22,
        "f1_pct": 33.33
    },
    {
        "class_index": 20,
        "class_name": "Tomato_leaf_bacterial_spot",
        "species": "Tomato",
        "condition": "Bacterial spot (Xanthomonas)",
        "visual_health": "Diseased",
        "quality_category": "Poor Quality (Visually Diseased/Damaged)",
        "support": 17,
        "true_positives": 2,
        "false_positives": 3,
        "false_negatives": 15,
        "precision": 0.4,
        "recall": 0.1176,
        "f1_score": 0.1818,
        "precision_pct": 40.0,
        "recall_pct": 11.76,
        "f1_pct": 18.18
    },
    {
        "class_index": 21,
        "class_name": "Tomato_leaf_late_blight",
        "species": "Tomato",
        "condition": "Late blight (Phytophthora infestans)",
        "visual_health": "Diseased",
        "quality_category": "Poor Quality (Visually Diseased/Damaged)",
        "support": 18,
        "true_positives": 7,
        "false_positives": 16,
        "false_negatives": 11,
        "precision": 0.3043,
        "recall": 0.3889,
        "f1_score": 0.3415,
        "precision_pct": 30.43,
        "recall_pct": 38.89,
        "f1_pct": 34.15
    },
    {
        "class_index": 22,
        "class_name": "Tomato_leaf_mosaic_virus",
        "species": "Tomato",
        "condition": "Tomato mosaic virus (ToMV)",
        "visual_health": "Diseased",
        "quality_category": "Poor Quality (Visually Diseased/Damaged)",
        "support": 9,
        "true_positives": 1,
        "false_positives": 3,
        "false_negatives": 8,
        "precision": 0.25,
        "recall": 0.1111,
        "f1_score": 0.1538,
        "precision_pct": 25.0,
        "recall_pct": 11.11,
        "f1_pct": 15.38
    },
    {
        "class_index": 23,
        "class_name": "Tomato_leaf_yellow_virus",
        "species": "Tomato",
        "condition": "Tomato yellow leaf curl virus (TYLCV)",
        "visual_health": "Diseased",
        "quality_category": "Poor Quality (Visually Diseased/Damaged)",
        "support": 37,
        "true_positives": 28,
        "false_positives": 9,
        "false_negatives": 9,
        "precision": 0.7568,
        "recall": 0.7568,
        "f1_score": 0.7568,
        "precision_pct": 75.68,
        "recall_pct": 75.68,
        "f1_pct": 75.68
    },
    {
        "class_index": 24,
        "class_name": "Tomato_mold_leaf",
        "species": "Tomato",
        "condition": "Leaf mold (Passalora fulva)",
        "visual_health": "Diseased",
        "quality_category": "Poor Quality (Visually Diseased/Damaged)",
        "support": 15,
        "true_positives": 7,
        "false_positives": 15,
        "false_negatives": 8,
        "precision": 0.3182,
        "recall": 0.4667,
        "f1_score": 0.3784,
        "precision_pct": 31.82,
        "recall_pct": 46.67,
        "f1_pct": 37.84
    },
    {
        "class_index": 25,
        "class_name": "Tomato_two_spotted_spider_mites_leaf",
        "species": "Tomato",
        "condition": "Two-spotted spider mites (Tetranychus urticae)",
        "visual_health": "Pest-Damaged",
        "quality_category": "Poor Quality (Visually Diseased/Damaged)",
        "support": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "precision_pct": 0.0,
        "recall_pct": 0.0,
        "f1_pct": 0.0
    },
    {
        "class_index": 26,
        "class_name": "grape_leaf",
        "species": "Grape",
        "condition": "Healthy foliage",
        "visual_health": "Healthy",
        "quality_category": "Good Quality (Visually Healthy)",
        "support": 12,
        "true_positives": 10,
        "false_positives": 8,
        "false_negatives": 2,
        "precision": 0.5556,
        "recall": 0.8333,
        "f1_score": 0.6667,
        "precision_pct": 55.56,
        "recall_pct": 83.33,
        "f1_pct": 66.67
    },
    {
        "class_index": 27,
        "class_name": "grape_leaf_black_rot",
        "species": "Grape",
        "condition": "Black rot (Guignardia bidwellii)",
        "visual_health": "Diseased",
        "quality_category": "Poor Quality (Visually Diseased/Damaged)",
        "support": 13,
        "true_positives": 10,
        "false_positives": 7,
        "false_negatives": 3,
        "precision": 0.5882,
        "recall": 0.7692,
        "f1_score": 0.6667,
        "precision_pct": 58.82,
        "recall_pct": 76.92,
        "f1_pct": 66.67
    }
];

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

    function init() {
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
        if (evalClassTableBody && (!evalClassTableBody.children || evalClassTableBody.children.length === 0)) {
            renderClassTable(allClassMetrics);
        }

        // Bind event listeners
        if (evalClassSearchInput) {
            evalClassSearchInput.addEventListener('input', handleClassSearch);
        }

        if (btnRetryMatrixLoad) {
            btnRetryMatrixLoad.addEventListener('click', loadConfusionMatrix);
        }

        if (btnZoomMatrixModal) {
            btnZoomMatrixModal.addEventListener('click', openMatrixModal);
        }

        if (matrixImageWrapper) {
            matrixImageWrapper.addEventListener('click', openMatrixModal);
        }

        if (btnCloseMatrixModal) {
            btnCloseMatrixModal.addEventListener('click', closeMatrixModal);
        }

        if (confusionMatrixModal) {
            confusionMatrixModal.addEventListener('click', (e) => {
                if (e.target === confusionMatrixModal) {
                    closeMatrixModal();
                }
            });
        }

        // Close on ESC
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && confusionMatrixModal && confusionMatrixModal.classList.contains('active')) {
                closeMatrixModal();
            }
        });

        // Asynchronously synchronize with live backend
        loadModelEvaluation();
        loadConfusionMatrix();
    }

    async function loadModelEvaluation() {
        try {
            let data = null;
            if (typeof ApiService !== 'undefined' && typeof ApiService.getModelEvaluation === 'function') {
                data = await ApiService.getModelEvaluation();
            } else {
                const fallbackUrl = (typeof ApiService !== 'undefined' && ApiService.BASE_URL)
                    ? `${ApiService.BASE_URL}/api/model/evaluation`
                    : '/api/model/evaluation';
                const resp = await fetch(fallbackUrl, { headers: { 'Accept': 'application/json' } });
                if (resp.ok) data = await resp.json();
            }

            if (data && data.status === 'SUCCESS' && Array.isArray(data.per_class_metrics) && data.per_class_metrics.length > 0) {
                evaluationData = data;
                allClassMetrics = data.per_class_metrics;
                renderOverallMetrics(data.overall_metrics, data.dataset_info);
                renderClassTable(allClassMetrics);
                console.log(`[ModelEvaluationController] Live metrics loaded: ${allClassMetrics.length} classes.`);
                return;
            }
        } catch (error) {
            console.warn('[ModelEvaluationController] Live evaluation metrics fetch failed, using empirical baseline:', error.message);
        }

        // Fallback: Ensure empirical data is rendered
        renderOverallMetrics(FALLBACK_OVERALL_METRICS, FALLBACK_DATASET_INFO);
        renderClassTable(allClassMetrics);
    }

    function renderOverallMetrics(overall, dataset) {
        if (!overall) return;

        if (evalAccuracy) {
            evalAccuracy.textContent = `${overall.accuracy_percent !== undefined ? overall.accuracy_percent.toFixed(2) : '56.56'}%`;
        }
        if (evalLoss) {
            evalLoss.textContent = overall.loss !== undefined ? overall.loss.toFixed(4) : '1.5560';
        }
        if (evalMacroF1) {
            evalMacroF1.textContent = `${overall.macro_f1_percent !== undefined ? overall.macro_f1_percent.toFixed(2) : '49.74'}%`;
        }
        if (evalWeightedF1) {
            evalWeightedF1.textContent = `${overall.weighted_f1_percent !== undefined ? overall.weighted_f1_percent.toFixed(2) : '53.80'}%`;
        }
        if (evalTestSamples) {
            evalTestSamples.textContent = overall.total_test_samples !== undefined ? overall.total_test_samples.toString() : '465';
        }
        if (evalClasses) {
            evalClasses.textContent = (dataset && (dataset.classes_supported || dataset.total_classes)) ? (dataset.classes_supported || dataset.total_classes).toString() : '28';
        }
    }

    function renderClassTable(classes) {
        if (!evalClassTableBody) return;

        if (!classes || classes.length === 0) {
            evalClassTableBody.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center py-4 text-muted">
                        <i class="fas fa-search" style="margin-right:6px;"></i> No class evaluation metrics match your search query.
                    </td>
                </tr>
            `;
            return;
        }

        let html = '';
        classes.forEach(c => {
            const vh = c.visual_health || 'Diseased';
            const isHealthy = vh === 'Healthy';
            let healthBadge = '';

            if (isHealthy) {
                healthBadge = '<span class="status-badge badge-healthy" style="padding:2px 8px;font-size:0.75rem;"><i class="fas fa-heart"></i> Healthy</span>';
            } else if (vh === 'Diseased') {
                healthBadge = '<span class="status-badge badge-diseased" style="padding:2px 8px;font-size:0.75rem;"><i class="fas fa-virus"></i> Diseased</span>';
            } else {
                healthBadge = `<span class="status-badge" style="background:rgba(217,119,6,0.1);color:#d97706;border:1px solid rgba(217,119,6,0.3);padding:2px 8px;font-size:0.75rem;"><i class="fas fa-bug"></i> ${escapeHtml(vh)}</span>`;
            }

            const precVal = typeof c.precision === 'number' ? c.precision : 0;
            const recVal = typeof c.recall === 'number' ? c.recall : 0;
            const f1Val = typeof c.f1_score === 'number' ? c.f1_score : 0;
            const supVal = typeof c.support === 'number' ? c.support : 0;

            const precStr = (precVal * 100).toFixed(1) + '%';
            const recStr = (recVal * 100).toFixed(1) + '%';
            const f1Pct = (f1Val * 100).toFixed(1);

            let barColor = '#16a34a';
            if (f1Val < 0.40) {
                barColor = '#dc2626';
            } else if (f1Val < 0.60) {
                barColor = '#d97706';
            }

            const cleanName = escapeHtml((c.class_name || '').replace(/_/g, ' '));
            const cleanCond = escapeHtml(c.condition || cleanName);
            const speciesVal = escapeHtml(c.species || '');

            html += `
                <tr data-class-name="${cleanName.toLowerCase()}" data-species="${speciesVal.toLowerCase()}" data-health="${vh.toLowerCase()}">
                    <td class="eval-col-name">
                        <span style="font-weight:700;color:var(--text-primary);font-size:0.88rem;">${cleanName}</span>
                        <span style="font-size:0.78rem;color:var(--text-muted);font-weight:400;margin-left:6px;">&bull; ${cleanCond}</span>
                    </td>
                    <td class="eval-col-health">${healthBadge}</td>
                    <td class="eval-col-stat" style="font-size:0.88rem;font-weight:600;">${precStr}</td>
                    <td class="eval-col-stat" style="font-size:0.88rem;font-weight:600;">${recStr}</td>
                    <td class="eval-col-f1">
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span style="font-size:0.85rem;font-weight:700;color:var(--text-primary);min-width:42px;">${f1Pct}%</span>
                            <div style="flex:1;background:#e2e8f0;height:6px;border-radius:3px;overflow:hidden;min-width:60px;">
                                <div style="width:${Math.min(100, Math.max(0, parseFloat(f1Pct)))}%;background:${barColor};height:100%;border-radius:3px;"></div>
                            </div>
                        </div>
                    </td>
                    <td style="font-size:0.88rem;font-weight:700;text-align:center;">${supVal}</td>
                </tr>
            `;
        });

        evalClassTableBody.innerHTML = html;
    }

    function handleClassSearch() {
        if (!evalClassSearchInput) return;
        const query = evalClassSearchInput.value.toLowerCase().trim();

        if (!query) {
            renderClassTable(allClassMetrics);
            return;
        }

        const filtered = allClassMetrics.filter(c => {
            const nameMatch = c.class_name && c.class_name.toLowerCase().includes(query);
            const speciesMatch = c.species && c.species.toLowerCase().includes(query);
            const conditionMatch = c.condition && c.condition.toLowerCase().includes(query);
            const healthMatch = c.visual_health && c.visual_health.toLowerCase().includes(query);
            return nameMatch || speciesMatch || conditionMatch || healthMatch;
        });

        renderClassTable(filtered);
    }

    /**
     * Loads the 28x28 confusion matrix image with loading states and fallback recovery.
     */
    function loadConfusionMatrix() {
        const matrixUrl = (typeof ApiService !== 'undefined' && typeof ApiService.getConfusionMatrixUrl === 'function')
            ? ApiService.getConfusionMatrixUrl()
            : '/api/model/confusion-matrix';
        const fallbackUrl = (typeof ApiService !== 'undefined' && ApiService.BASE_URL)
            ? `${ApiService.BASE_URL}/assets/confusion_matrix.png`
            : 'assets/confusion_matrix.png';

        console.log('[ModelEvaluationController] Requesting Confusion Matrix from:', matrixUrl);

        if (btnConfusionMatrixFull) {
            btnConfusionMatrixFull.href = matrixUrl;
        }
        if (modalRawPngLink) {
            modalRawPngLink.href = matrixUrl;
        }

        if (!evalConfusionMatrixImg) return;

        // Show spinner, hide error and image
        if (matrixLoadingSpinner) matrixLoadingSpinner.style.display = 'flex';
        if (matrixErrorContainer) matrixErrorContainer.style.display = 'none';
        if (matrixImageWrapper) matrixImageWrapper.style.display = 'none';

        const testImg = new Image();
        testImg.onload = function() {
            evalConfusionMatrixImg.src = matrixUrl;
            if (modalConfusionMatrixImg) modalConfusionMatrixImg.src = matrixUrl;
            if (matrixLoadingSpinner) matrixLoadingSpinner.style.display = 'none';
            if (matrixErrorContainer) matrixErrorContainer.style.display = 'none';
            if (matrixImageWrapper) matrixImageWrapper.style.display = 'block';
            console.log('[ModelEvaluationController] Confusion Matrix successfully loaded.');
        };

        testImg.onerror = function() {
            console.warn('[ModelEvaluationController] Primary endpoint failed, attempting fallback to static asset:', fallbackUrl);
            const fallbackImg = new Image();
            fallbackImg.onload = function() {
                evalConfusionMatrixImg.src = fallbackUrl;
                if (modalConfusionMatrixImg) modalConfusionMatrixImg.src = fallbackUrl;
                if (btnConfusionMatrixFull) btnConfusionMatrixFull.href = fallbackUrl;
                if (modalRawPngLink) modalRawPngLink.href = fallbackUrl;
                if (matrixLoadingSpinner) matrixLoadingSpinner.style.display = 'none';
                if (matrixErrorContainer) matrixErrorContainer.style.display = 'none';
                if (matrixImageWrapper) matrixImageWrapper.style.display = 'block';
                console.log('[ModelEvaluationController] Confusion Matrix loaded via fallback asset route.');
            };

            fallbackImg.onerror = function(err) {
                console.error('[ModelEvaluationController] Failed to load confusion matrix from both endpoints:', err);
                if (matrixLoadingSpinner) matrixLoadingSpinner.style.display = 'none';
                if (matrixImageWrapper) matrixImageWrapper.style.display = 'none';
                if (matrixErrorContainer) {
                    matrixErrorContainer.style.display = 'block';
                    if (matrixErrorText) {
                        matrixErrorText.textContent = `Could not reach ${matrixUrl}. Ensure the Flask server is running at http://127.0.0.1:5000.`;
                    }
                }
            };
            fallbackImg.src = fallbackUrl;
        };

        testImg.src = matrixUrl;
    }

    function openMatrixModal() {
        if (!confusionMatrixModal) return;
        const currentSrc = evalConfusionMatrixImg && evalConfusionMatrixImg.src
            ? evalConfusionMatrixImg.src
            : ((typeof ApiService !== 'undefined' && typeof ApiService.getConfusionMatrixUrl === 'function')
                ? ApiService.getConfusionMatrixUrl()
                : '/api/model/confusion-matrix');

        if (modalConfusionMatrixImg) {
            modalConfusionMatrixImg.src = currentSrc;
        }
        confusionMatrixModal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    function closeMatrixModal() {
        if (!confusionMatrixModal) return;
        confusionMatrixModal.classList.remove('active');
        document.body.style.overflow = '';
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    return {
        init,
        loadModelEvaluation,
        loadConfusionMatrix,
        openMatrixModal,
        closeMatrixModal
    };
})();

// Explicitly attach to window
window.ModelEvaluationController = ModelEvaluationController;
