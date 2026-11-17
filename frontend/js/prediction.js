/**
 * Leaf Quality Classification and Plucking Standard Monitoring System
 * Module 7: Prediction Results & UI Presentation Controller
 * File: frontend/js/prediction.js
 */

const PredictionController = (function() {
    // DOM Elements
    let btnAnalyze, btnResetAnalysis;
    let loadingOverlay, resultsCard, analysisDashboard;
    let resultStatusBadge, diagnosisBanner, predictedSpecies, predictedCondition;
    let confidenceGaugeVal, blurAlertBox, blurAlertText, uncertaintyAlertBox, uncertaintyAlertText;
    let metricSharpness, metricLaplacian, metricBrightness;
    let topCandidatesList, recommendationBox, recHeader, recBody, disclaimerText;
    let thresholdSlider, thresholdVal;

    function init() {
        btnAnalyze = document.getElementById('btnAnalyzeLeaf');
        btnResetAnalysis = document.getElementById('btnResetAnalysis');
        loadingOverlay = document.getElementById('loadingOverlay');
        resultsCard = document.getElementById('resultsCard');
        analysisDashboard = document.getElementById('analysisDashboard');

        resultStatusBadge = document.getElementById('resultStatusBadge');
        diagnosisBanner = document.getElementById('diagnosisBanner');
        predictedSpecies = document.getElementById('predictedSpecies');
        predictedCondition = document.getElementById('predictedCondition');
        confidenceGaugeVal = document.getElementById('confidenceGaugeVal');

        blurAlertBox = document.getElementById('blurAlertBox');
        blurAlertText = document.getElementById('blurAlertText');
        uncertaintyAlertBox = document.getElementById('uncertaintyAlertBox');
        uncertaintyAlertText = document.getElementById('uncertaintyAlertText');

        metricSharpness = document.getElementById('metricSharpness');
        metricLaplacian = document.getElementById('metricLaplacian');
        metricBrightness = document.getElementById('metricBrightness');

        topCandidatesList = document.getElementById('topCandidatesList');
        recommendationBox = document.getElementById('recommendationBox');
        recHeader = document.getElementById('recHeader');
        recBody = document.getElementById('recBody');
        disclaimerText = document.getElementById('disclaimerText');

        thresholdSlider = document.getElementById('thresholdSlider');
        thresholdVal = document.getElementById('thresholdVal');

        setupEventListeners();
    }

    function setupEventListeners() {
        if (btnAnalyze) {
            btnAnalyze.addEventListener('click', handleAnalyzeClick);
        }
        if (btnResetAnalysis) {
            btnResetAnalysis.addEventListener('click', () => {
                UploadController.resetSelection();
            });
        }
        if (thresholdSlider) {
            thresholdSlider.addEventListener('input', (e) => {
                const val = (parseFloat(e.target.value) * 100).toFixed(0);
                thresholdVal.textContent = `${val}%`;
            });
        }
    }

    async function handleAnalyzeClick() {
        const fileBlob = UploadController.getCurrentBlob();
        const filename = UploadController.getCurrentFilename();

        if (!fileBlob) {
            App.showToast('Please upload or capture a leaf photo first.', 'warning');
            return;
        }

        const threshold = thresholdSlider ? parseFloat(thresholdSlider.value) : 0.40;

        // Show loading state
        setLoading(true);

        try {
            const result = await ApiService.predictLeaf(fileBlob, filename, threshold);
            renderPredictionResults(result);
            App.showToast('Analysis completed successfully.', 'info');
            if (window.HistoryController && typeof HistoryController.refresh === 'function') {
                HistoryController.refresh();
            }
        } catch (error) {
            console.error('[PredictionController] Error during analysis:', error);
            App.showToast(error.message || 'Failed to analyze leaf image.', 'error');
        } finally {
            setLoading(false);
        }
    }

    function setLoading(isLoading) {
        if (isLoading) {
            loadingOverlay.classList.add('active');
            btnAnalyze.disabled = true;
            btnAnalyze.innerHTML = '<span class="spinner" style="width:16px;height:16px;border-width:2px;"></span> Analyzing...';
        } else {
            loadingOverlay.classList.remove('active');
            btnAnalyze.disabled = false;
            btnAnalyze.innerHTML = '<i class="fas fa-microscope"></i> Analyze Leaf';
        }
    }

    function renderPredictionResults(data) {
        // Expand dashboard layout to 2 columns
        analysisDashboard.classList.remove('results-hidden');
        resultsCard.classList.add('active');

        const isSuccess = data.status === "SUCCESS";
        const isHealthy = data.is_visually_healthy;

        // 1. Status Badge
        resultStatusBadge.className = 'result-header-badge ' + (isSuccess ? 'badge-success' : 'badge-uncertain');
        resultStatusBadge.innerHTML = isSuccess 
            ? '<i class="fas fa-check-circle"></i> Certain Classification'
            : '<i class="fas fa-exclamation-triangle"></i> Uncertain / Low Confidence';

        // 2. Diagnosis Banner Styling
        diagnosisBanner.className = 'diagnosis-banner ' + (isHealthy ? '' : 'diseased');
        if (!isSuccess) {
            diagnosisBanner.classList.add('uncertain');
        }

        predictedSpecies.textContent = data.species || 'Unknown Plant';
        predictedCondition.textContent = data.condition || 'Condition Unconfirmed';
        confidenceGaugeVal.textContent = `${data.confidence_percent.toFixed(1)}%`;

        // 3. Image Quality Metrics
        const quality = data.quality_assessment || {};
        metricSharpness.textContent = quality.quality_rating || 'Normal';
        metricLaplacian.textContent = quality.laplacian_variance !== undefined ? quality.laplacian_variance : 'N/A';
        metricBrightness.textContent = quality.mean_brightness !== undefined ? `${quality.mean_brightness.toFixed(0)}/255` : 'N/A';

        // 4. Blur Alert
        if (data.blur_warning) {
            blurAlertBox.style.display = 'flex';
            blurAlertText.textContent = data.blur_warning;
        } else {
            blurAlertBox.style.display = 'none';
        }

        // 5. Uncertainty Alert
        if (data.confidence_warning) {
            uncertaintyAlertBox.style.display = 'flex';
            uncertaintyAlertText.textContent = data.confidence_warning;
        } else {
            uncertaintyAlertBox.style.display = 'none';
        }

        // 6. Top-3 Candidate Progress Bars
        renderTopCandidates(data.top_three || []);

        // 7. Agronomic Plucking Recommendation
        renderPluckingRecommendation(data);

        // 8. Disclaimer
        if (disclaimerText) {
            disclaimerText.textContent = data.disclaimer || '';
        }

        // Smooth scroll to results on mobile devices
        if (window.innerWidth <= 992) {
            resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    function renderTopCandidates(candidates) {
        topCandidatesList.innerHTML = '';

        if (!candidates || candidates.length === 0) {
            topCandidatesList.innerHTML = '<p class="text-muted">No candidate scores available.</p>';
            return;
        }

        candidates.forEach(cand => {
            const isCandHealthy = cand.visual_health_type === "Healthy";
            const typeClass = isCandHealthy ? "type-healthy" : "type-diseased";
            const fillClass = isCandHealthy ? "fill-healthy" : "fill-diseased";
            const cleanName = cand.class_name.replace(/_/g, ' ');

            const itemHtml = `
                <div class="candidate-item">
                    <div class="candidate-meta">
                        <span class="candidate-name">#${cand.rank} ${cleanName}</span>
                        <div>
                            <span class="candidate-type-badge ${typeClass}">${cand.visual_health_type}</span>
                            <strong style="margin-left:6px;">${cand.confidence_percent.toFixed(1)}%</strong>
                        </div>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill ${fillClass}" style="width: 0%" data-target="${cand.confidence_percent}%"></div>
                    </div>
                </div>
            `;
            topCandidatesList.insertAdjacentHTML('beforeend', itemHtml);
        });

        // Trigger CSS transition animation after DOM injection
        setTimeout(() => {
            const fills = topCandidatesList.querySelectorAll('.progress-fill');
            fills.forEach(fill => {
                fill.style.width = fill.getAttribute('data-target');
            });
        }, 50);
    }

    function renderPluckingRecommendation(data) {
        const isHealthy = data.is_visually_healthy;
        const isCertain = data.status === "SUCCESS";

        if (!isCertain) {
            recommendationBox.className = 'recommendation-box uncertain';
            recHeader.innerHTML = '<i class="fas fa-question-circle"></i> Harvest Decision Deferred';
            recBody.textContent = data.plucking_recommendation;
        } else if (isHealthy) {
            recommendationBox.className = 'recommendation-box';
            recHeader.innerHTML = '<i class="fas fa-hand-holding-seedling"></i> Harvest Assessment: Conditional Field Verification Required';
            recBody.textContent = data.plucking_recommendation;
        } else {
            recommendationBox.className = 'recommendation-box reject';
            recHeader.innerHTML = '<i class="fas fa-ban"></i> Harvest Rejection: Do Not Pluck';
            recBody.textContent = data.plucking_recommendation;
        }
    }

    function hideResults() {
        if (resultsCard) {
            resultsCard.classList.remove('active');
        }
        if (analysisDashboard) {
            analysisDashboard.classList.add('results-hidden');
        }
    }

    return {
        init,
        hideResults
    };
})();
