/**
 * Leaf Quality Classification and Plucking Standard Monitoring System
 * Module 8: Historical Analysis & Database View Controller
 * File: frontend/js/history.js
 */

const HistoryController = (function() {
    // State
    let currentPage = 1;
    const pageSize = 10;
    let totalPages = 1;
    let searchDebounceTimer = null;

    // DOM Elements
    let statTotalAnalyses, statHealthyCount, statDiseasedCount, statAvgConfidence;
    let historyTableBody, historyTableContainer, historyEmptyState, historyLoadingOverlay;
    let historyPagination, btnPrevPage, btnNextPage, historyPageIndicator;
    let historyFilterSpecies, historyFilterHealth, historySearchInput, btnClearSearch;
    let btnClearAllHistory;
    let historyDetailModal, btnCloseModal, modalRecordDetails;

    function init() {
        console.log('[HistoryController] Initializing History & Analytics UI...');

        // Cache DOM elements
        statTotalAnalyses = document.getElementById('statTotalAnalyses');
        statHealthyCount = document.getElementById('statHealthyCount');
        statDiseasedCount = document.getElementById('statDiseasedCount');
        statAvgConfidence = document.getElementById('statAvgConfidence');

        historyTableBody = document.getElementById('historyTableBody');
        historyTableContainer = document.getElementById('historyTableContainer');
        historyEmptyState = document.getElementById('historyEmptyState');
        historyLoadingOverlay = document.getElementById('historyLoadingOverlay');

        historyPagination = document.getElementById('historyPagination');
        btnPrevPage = document.getElementById('btnPrevPage');
        btnNextPage = document.getElementById('btnNextPage');
        historyPageIndicator = document.getElementById('historyPageIndicator');

        historyFilterSpecies = document.getElementById('historyFilterSpecies');
        historyFilterHealth = document.getElementById('historyFilterHealth');
        historySearchInput = document.getElementById('historySearchInput');
        btnClearSearch = document.getElementById('btnClearSearch');

        btnClearAllHistory = document.getElementById('btnClearAllHistory');

        historyDetailModal = document.getElementById('historyDetailModal');
        btnCloseModal = document.getElementById('btnCloseModal');
        modalRecordDetails = document.getElementById('modalRecordDetails');

        setupEventListeners();

        // Initial fetch of statistics and records
        loadStats();
        loadHistory(1);
    }

    function setupEventListeners() {
        // Search input with debounce (300ms)
        if (historySearchInput) {
            historySearchInput.addEventListener('input', () => {
                if (searchDebounceTimer) clearTimeout(searchDebounceTimer);
                searchDebounceTimer = setTimeout(() => {
                    currentPage = 1;
                    loadHistory(1);
                }, 300);
            });
        }

        // Clear search button
        if (btnClearSearch && historySearchInput) {
            btnClearSearch.addEventListener('click', () => {
                historySearchInput.value = '';
                currentPage = 1;
                loadHistory(1);
            });
        }

        // Dropdown filters
        if (historyFilterSpecies) {
            historyFilterSpecies.addEventListener('change', () => {
                currentPage = 1;
                loadHistory(1);
            });
        }

        if (historyFilterHealth) {
            historyFilterHealth.addEventListener('change', () => {
                currentPage = 1;
                loadHistory(1);
            });
        }

        // Pagination buttons
        if (btnPrevPage) {
            btnPrevPage.addEventListener('click', () => {
                if (currentPage > 1) {
                    loadHistory(currentPage - 1);
                }
            });
        }

        if (btnNextPage) {
            btnNextPage.addEventListener('click', () => {
                if (currentPage < totalPages) {
                    loadHistory(currentPage + 1);
                }
            });
        }

        // Clear All History button
        if (btnClearAllHistory) {
            btnClearAllHistory.addEventListener('click', handleClearAllHistory);
        }

        // Modal close button and backdrop click
        if (btnCloseModal && historyDetailModal) {
            btnCloseModal.addEventListener('click', closeModal);
            historyDetailModal.addEventListener('click', (e) => {
                if (e.target === historyDetailModal) {
                    closeModal();
                }
            });
        }

        // Keyboard ESC to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && historyDetailModal && historyDetailModal.classList.contains('active')) {
                closeModal();
            }
        });
    }

    /**
     * Loads aggregated analytics from /api/history/stats.
     */
    async function loadStats() {
        try {
            const data = await ApiService.getHistoryStats();
            if (data.status === 'SUCCESS' && data.statistics) {
                renderStats(data.statistics);
            }
        } catch (error) {
            console.warn('[HistoryController] Failed to fetch stats:', error.message);
        }
    }

    function renderStats(stats) {
        if (statTotalAnalyses) {
            statTotalAnalyses.textContent = stats.total_records !== undefined ? stats.total_records.toLocaleString() : '0';
        }
        if (statHealthyCount) {
            statHealthyCount.textContent = stats.healthy_count !== undefined ? stats.healthy_count.toLocaleString() : '0';
        }
        if (statDiseasedCount) {
            statDiseasedCount.textContent = stats.diseased_count !== undefined ? stats.diseased_count.toLocaleString() : '0';
        }
        if (statAvgConfidence) {
            statAvgConfidence.textContent = stats.average_confidence_percent !== undefined
                ? `${stats.average_confidence_percent.toFixed(1)}%`
                : '0.0%';
        }

        // Dynamically populate species filter options if empty
        if (historyFilterSpecies && stats.by_species) {
            const currentSelected = historyFilterSpecies.value;
            // Retain the first option "All Plant Species"
            const firstOpt = historyFilterSpecies.options[0];
            historyFilterSpecies.innerHTML = '';
            if (firstOpt) historyFilterSpecies.appendChild(firstOpt);

            const speciesList = Object.keys(stats.by_species).sort();
            speciesList.forEach(sp => {
                const opt = document.createElement('option');
                opt.value = sp;
                opt.textContent = `${sp} (${stats.by_species[sp]})`;
                if (sp === currentSelected) {
                    opt.selected = true;
                }
                historyFilterSpecies.appendChild(opt);
            });
        }
    }

    /**
     * Fetches and renders paginated history records.
     */
    async function loadHistory(page = 1) {
        currentPage = page;
        setTableLoading(true);

        const params = {
            page: page,
            limit: pageSize
        };

        if (historySearchInput && historySearchInput.value.trim()) {
            params.search = historySearchInput.value.trim();
        }

        if (historyFilterSpecies && historyFilterSpecies.value) {
            params.species = historyFilterSpecies.value;
        }

        if (historyFilterHealth && historyFilterHealth.value) {
            params.visual_health = historyFilterHealth.value;
        }

        try {
            const data = await ApiService.getHistory(params);
            if (data.status === 'SUCCESS') {
                renderTable(data.records || [], data.pagination || {});
            }
        } catch (error) {
            console.error('[HistoryController] Failed to load history:', error);
            if (historyTableBody) {
                historyTableBody.innerHTML = `
                    <tr>
                        <td colspan="7" class="text-center py-4 text-muted">
                            <i class="fas fa-exclamation-triangle" style="color:var(--warning);margin-right:8px;"></i>
                            Failed to load analysis history: ${error.message}
                        </td>
                    </tr>
                `;
            }
        } finally {
            setTableLoading(false);
        }
    }

    function setTableLoading(isLoading) {
        if (historyLoadingOverlay) {
            if (isLoading) {
                historyLoadingOverlay.classList.add('active');
            } else {
                historyLoadingOverlay.classList.remove('active');
            }
        }
    }

    function renderTable(records, pagination) {
        if (!historyTableBody) return;

        totalPages = pagination.total_pages || 1;
        currentPage = pagination.page || 1;

        // Update pagination UI
        if (historyPageIndicator) {
            const total = pagination.total_records || 0;
            historyPageIndicator.textContent = `Page ${currentPage} of ${totalPages} (${total} total)`;
        }

        if (btnPrevPage) {
            btnPrevPage.disabled = !pagination.has_prev;
        }
        if (btnNextPage) {
            btnNextPage.disabled = !pagination.has_next;
        }

        if (records.length === 0) {
            historyTableBody.innerHTML = '';
            if (historyEmptyState) historyEmptyState.style.display = 'block';
            if (historyTableContainer) historyTableContainer.style.display = 'none';
            if (historyPagination) historyPagination.style.display = 'none';
            return;
        }

        if (historyEmptyState) historyEmptyState.style.display = 'none';
        if (historyTableContainer) historyTableContainer.style.display = 'block';
        if (historyPagination) historyPagination.style.display = 'flex';

        let html = '';
        records.forEach(rec => {
            const formattedTime = formatTimestamp(rec.timestamp);
            const isHealthy = rec.visual_health === 'Healthy';
            const healthBadgeClass = isHealthy ? 'badge-healthy' : 'badge-diseased';
            const isCertain = rec.prediction_status === 'Certain' || rec.prediction_status === 'SUCCESS';
            const statusIcon = isCertain
                ? '<i class="fas fa-check-circle" style="color:var(--primary);margin-right:4px;" title="Certain Prediction"></i>'
                : '<i class="fas fa-exclamation-triangle" style="color:var(--warning);margin-right:4px;" title="Uncertain / Low Confidence"></i>';

            let suitBadgeClass = 'badge-suit-reject';
            let suitText = 'Do Not Pluck';
            const rawSuit = String(rec.plucking_suitability || '').toUpperCase();
            if (rawSuit.includes('COND')) {
                suitBadgeClass = 'badge-suit-conditional';
                suitText = 'Conditional';
            } else if (rawSuit.includes('DEF')) {
                suitBadgeClass = 'badge-suit-deferred';
                suitText = 'Deferred';
            }

            const cleanClass = (rec.predicted_class || 'Unknown').replace(/_/g, ' ');

            html += `
                <tr data-record-id="${rec.id}">
                    <td class="col-id">#${rec.id}</td>
                    <td class="col-time" title="${rec.timestamp || ''}">
                        <div class="time-primary">${formattedTime.date}</div>
                        <div class="time-secondary">${formattedTime.time}</div>
                    </td>
                    <td class="col-species">
                        <strong>${escapeHtml(rec.species || 'Unknown')}</strong>
                        <div class="crop-filename" title="${escapeHtml(rec.image_filename || '')}">${escapeHtml(rec.image_filename || 'capture.jpg')}</div>
                    </td>
                    <td class="col-class">
                        <span class="class-label">${escapeHtml(cleanClass)}</span>
                    </td>
                    <td class="col-confidence">
                        <div class="conf-cell">
                            ${statusIcon}
                            <span class="conf-num">${(rec.confidence_percent || 0).toFixed(1)}%</span>
                        </div>
                    </td>
                    <td class="col-health">
                        <span class="badge ${healthBadgeClass}">${rec.visual_health || 'Unknown'}</span>
                    </td>
                    <td class="col-suitability">
                        <span class="badge ${suitBadgeClass}">${suitText}</span>
                    </td>
                    <td class="col-actions">
                        <button class="btn-action-view" onclick="HistoryController.viewRecord(${rec.id})" title="View Complete Diagnostics">
                            <i class="fas fa-eye"></i>
                        </button>
                        <button class="btn-action-delete" onclick="HistoryController.deleteRecord(${rec.id})" title="Delete Record">
                            <i class="fas fa-trash-alt"></i>
                        </button>
                    </td>
                </tr>
            `;
        });

        historyTableBody.innerHTML = html;
    }

    /**
     * Formats ISO timestamp string to readable date & time.
     */
    function formatTimestamp(isoStr) {
        if (!isoStr) return { date: 'N/A', time: '' };
        try {
            const d = new Date(isoStr);
            if (isNaN(d.getTime())) return { date: isoStr, time: '' };
            const date = d.toLocaleDateString(undefined, {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
            const time = d.toLocaleTimeString(undefined, {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
            return { date, time };
        } catch (e) {
            return { date: isoStr, time: '' };
        }
    }

    /**
     * Views complete diagnostic details in a modal dialog.
     */
    async function viewRecord(id) {
        if (!historyDetailModal || !modalRecordDetails) return;

        openModal();
        modalRecordDetails.innerHTML = `
            <div class="text-center py-5">
                <span class="spinner" style="width:36px;height:36px;border-width:3px;"></span>
                <p class="mt-3 text-muted">Retrieving diagnostic record #${id}...</p>
            </div>
        `;

        try {
            const data = await ApiService.getHistoryRecord(id);
            if (data.status === 'SUCCESS' && data.record) {
                renderModalRecord(data.record);
            } else {
                modalRecordDetails.innerHTML = `<div class="alert alert-error">Record #${id} not found.</div>`;
            }
        } catch (error) {
            console.error('[HistoryController] Failed to view record:', error);
            modalRecordDetails.innerHTML = `<div class="alert alert-error">Error loading record: ${error.message}</div>`;
        }
    }

    function renderModalRecord(rec) {
        const isHealthy = rec.visual_health === 'Healthy';
        const formattedTime = formatTimestamp(rec.timestamp);
        const quality = rec.quality_assessment || {};
        const top3 = rec.top_three || [];

        let top3Html = '';
        top3.forEach(c => {
            const candHealthy = c.visual_health_type === 'Healthy';
            const typeClass = candHealthy ? 'type-healthy' : 'type-diseased';
            const fillClass = candHealthy ? 'fill-healthy' : 'fill-diseased';
            const clean = (c.class_name || '').replace(/_/g, ' ');

            top3Html += `
                <div class="candidate-item" style="margin-bottom:0.75rem;">
                    <div class="candidate-meta" style="font-size:0.88rem;">
                        <span>#${c.rank} <strong>${escapeHtml(clean)}</strong></span>
                        <div>
                            <span class="candidate-type-badge ${typeClass}">${c.visual_health_type}</span>
                            <strong style="margin-left:6px;">${c.confidence_percent.toFixed(1)}%</strong>
                        </div>
                    </div>
                    <div class="progress-track" style="height:6px;">
                        <div class="progress-fill ${fillClass}" style="width:${c.confidence_percent}%"></div>
                    </div>
                </div>
            `;
        });

        modalRecordDetails.innerHTML = `
            <div class="record-detail-grid">
                <div class="detail-summary-card ${isHealthy ? 'healthy' : 'diseased'}">
                    <div class="detail-badge-row">
                        <span class="badge ${isHealthy ? 'badge-healthy' : 'badge-diseased'}">${rec.visual_health}</span>
                        <span class="badge ${(rec.prediction_status === 'Certain' || rec.prediction_status === 'SUCCESS') ? 'badge-certain' : 'badge-uncertain'}">${(rec.prediction_status === 'Certain' || rec.prediction_status === 'SUCCESS') ? 'Certain' : 'Uncertain'}</span>
                        <span class="record-meta-id">ID #${rec.id}</span>
                    </div>
                    <h3 class="detail-species-title">${escapeHtml(rec.species || 'Unknown Plant')}</h3>
                    <p class="detail-condition-desc">${escapeHtml(rec.condition || 'General')}</p>
                    <div class="detail-confidence-display">
                        <span class="conf-lead">Confidence:</span>
                        <span class="conf-highlight">${(rec.confidence_percent || 0).toFixed(1)}%</span>
                    </div>
                </div>

                <div class="detail-meta-list">
                    <div class="meta-row">
                        <span class="meta-label"><i class="fas fa-calendar-alt"></i> Timestamp:</span>
                        <span class="meta-value">${formattedTime.date} ${formattedTime.time}</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-label"><i class="fas fa-file-image"></i> Image Source:</span>
                        <span class="meta-value font-mono">${escapeHtml(rec.image_filename || 'capture.jpg')}</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-label"><i class="fas fa-tag"></i> Predicted Class:</span>
                        <span class="meta-value">${escapeHtml((rec.predicted_class || '').replace(/_/g, ' '))}</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-label"><i class="fas fa-shield-alt"></i> Plucking Suitability:</span>
                        <span class="meta-value font-semibold">${escapeHtml(rec.plucking_suitability || 'N/A')}</span>
                    </div>
                </div>
            </div>

            <!-- Quality Diagnostics -->
            <div class="detail-section">
                <h4><i class="fas fa-camera"></i> Image Quality Assessment</h4>
                <div class="quality-grid">
                    <div class="quality-card">
                        <span class="quality-lbl">Rating</span>
                        <strong class="quality-val">${escapeHtml(quality.quality_rating || 'Normal')}</strong>
                    </div>
                    <div class="quality-card">
                        <span class="quality-lbl">Laplacian Variance</span>
                        <strong class="quality-val">${quality.laplacian_variance !== undefined ? quality.laplacian_variance : 'N/A'}</strong>
                    </div>
                    <div class="quality-card">
                        <span class="quality-lbl">Mean Brightness</span>
                        <strong class="quality-val">${quality.mean_brightness !== undefined ? quality.mean_brightness.toFixed(0) + '/255' : 'N/A'}</strong>
                    </div>
                </div>
            </div>

            <!-- Top 3 Candidates -->
            <div class="detail-section">
                <h4><i class="fas fa-list-ol"></i> Top-3 Candidate Predictions</h4>
                <div class="top-candidates-box">
                    ${top3Html || '<p class="text-muted">No candidate scores recorded.</p>'}
                </div>
            </div>

            <!-- Agronomic Recommendation -->
            <div class="detail-section">
                <h4><i class="fas fa-clipboard-check"></i> Agronomic Advisory</h4>
                <div class="modal-recommendation-box ${rec.plucking_suitability === 'CONDITIONAL_VERIFICATION' ? 'conditional' : (rec.plucking_suitability === 'DEFERRED' ? 'deferred' : 'reject')}">
                    <p>${escapeHtml(rec.plucking_recommendation || 'No recommendation available.')}</p>
                </div>
            </div>
        `;
    }

    /**
     * Deletes a single analysis record with confirmation.
     */
    async function deleteRecord(id) {
        const confirmed = confirm(`Are you sure you want to delete analysis record #${id}? This action cannot be undone.`);
        if (!confirmed) return;

        try {
            const data = await ApiService.deleteHistoryRecord(id);
            if (data.status === 'SUCCESS') {
                App.showToast(`Record #${id} was deleted successfully.`, 'info');
                // Refresh table & stats
                loadStats();
                loadHistory(currentPage);
            }
        } catch (error) {
            console.error('[HistoryController] Failed to delete record:', error);
            App.showToast(`Failed to delete record: ${error.message}`, 'error');
        }
    }

    /**
     * Clears all historical records with double confirmation safeguard.
     */
    async function handleClearAllHistory() {
        const confirmed = confirm(
            'WARNING: You are about to permanently delete ALL historical analysis records and reset database analytics.\n\nAre you sure you want to proceed?'
        );
        if (!confirmed) return;

        try {
            const data = await ApiService.clearAllHistory();
            if (data.status === 'SUCCESS') {
                App.showToast(`All ${data.deleted_count} historical records cleared successfully.`, 'info');
                loadStats();
                loadHistory(1);
            }
        } catch (error) {
            console.error('[HistoryController] Failed to clear history:', error);
            App.showToast(`Failed to clear history: ${error.message}`, 'error');
        }
    }

    function openModal() {
        if (historyDetailModal) {
            historyDetailModal.classList.add('active');
            document.body.style.overflow = 'hidden';
        }
    }

    function closeModal() {
        if (historyDetailModal) {
            historyDetailModal.classList.remove('active');
            document.body.style.overflow = '';
        }
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

    /**
     * Public refresh method called by PredictionController when a new analysis completes.
     */
    function refresh() {
        console.log('[HistoryController] Auto-refreshing history following new analysis...');
        loadStats();
        loadHistory(1);
    }

    return {
        init,
        loadHistory,
        loadStats,
        viewRecord,
        deleteRecord,
        refresh
    };
})();
