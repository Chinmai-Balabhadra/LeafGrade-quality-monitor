/**
 * Leaf Quality Classification and Plucking Standard Monitoring System
 * Module 7: Backend API Communication Service
 * File: frontend/js/api.js
 */

const ApiService = (function() {
    // Dynamic Base URL detection:
    // If served from Flask port 5000, use relative paths; otherwise use http://127.0.0.1:5000
    const isFlaskOrigin = (window.location.protocol === 'http:' || window.location.protocol === 'https:') &&
                          (window.location.port === '5000' || window.location.hostname === 'localhost' && window.location.port === '5000');
    
    const BASE_URL = isFlaskOrigin ? '' : 'http://127.0.0.1:5000';

    /**
     * Checks the health and availability of the Flask backend and MobileNetV2 model.
     * @returns {Promise<Object>} Health status response payload.
     */
    async function checkHealth() {
        const url = `${BASE_URL}/api/health`;
        try {
            const response = await fetch(url, {
                method: 'GET',
                headers: { 'Accept': 'application/json' }
            });
            if (!response.ok) {
                throw new Error(`Health check returned HTTP ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.warn('[ApiService] Health check failed:', error.message);
            return {
                status: 'unreachable',
                service: 'Leaf Quality API',
                error: error.message,
                model_status: { model_ready: false }
            };
        }
    }

    /**
     * Uploads a leaf image (file or blob) for classification and plucking evaluation.
     * @param {File|Blob} fileData - The raw image file or canvas blob.
     * @param {string} [filename='leaf_capture.jpg'] - Filename for the multipart payload.
     * @param {number} [threshold=null] - Optional confidence threshold (0.0 to 1.0).
     * @returns {Promise<Object>} The structured prediction result.
     */
    async function predictLeaf(fileData, filename = 'leaf_capture.jpg', threshold = null) {
        const url = `${BASE_URL}/api/predict`;
        const formData = new FormData();

        // Ensure field name is strictly 'file' as required by API contract
        if (fileData instanceof File) {
            formData.append('file', fileData);
        } else {
            formData.append('file', fileData, filename);
        }

        if (threshold !== null && !isNaN(threshold)) {
            formData.append('confidence_threshold', threshold.toString());
        }

        try {
            const response = await fetch(url, {
                method: 'POST',
                body: formData,
                headers: { 'Accept': 'application/json' }
            });

            const data = await response.json();

            if (!response.ok) {
                const errorMsg = data.message || `Server error (HTTP ${response.status})`;
                throw new Error(errorMsg);
            }

            return data;
        } catch (error) {
            console.error('[ApiService] Prediction request failed:', error);
            throw error;
        }
    }

    /**
     * Retrieves paginated analysis history records with optional filters.
     * @param {Object} [params={}] - Query parameters (page, limit, species, visual_health, status, search)
     * @returns {Promise<Object>} History records and pagination metadata.
     */
    async function getHistory(params = {}) {
        const query = new URLSearchParams();
        Object.entries(params).forEach(([key, value]) => {
            if (value !== undefined && value !== null && value !== '') {
                query.append(key, value);
            }
        });

        const queryString = query.toString();
        const url = `${BASE_URL}/api/history${queryString ? '?' + queryString : ''}`;

        try {
            const response = await fetch(url, {
                method: 'GET',
                headers: { 'Accept': 'application/json' }
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.message || `HTTP ${response.status}`);
            }
            return data;
        } catch (error) {
            console.error('[ApiService] getHistory failed:', error);
            throw error;
        }
    }

    /**
     * Retrieves aggregated statistics for past leaf analyses.
     * @returns {Promise<Object>} Statistics summary.
     */
    async function getHistoryStats() {
        const url = `${BASE_URL}/api/history/stats`;
        try {
            const response = await fetch(url, {
                method: 'GET',
                headers: { 'Accept': 'application/json' }
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.message || `HTTP ${response.status}`);
            }
            return data;
        } catch (error) {
            console.error('[ApiService] getHistoryStats failed:', error);
            throw error;
        }
    }

    /**
     * Retrieves a single analysis record by ID.
     * @param {number|string} id - Record ID.
     * @returns {Promise<Object>} Full diagnostic record.
     */
    async function getHistoryRecord(id) {
        const url = `${BASE_URL}/api/history/${id}`;
        try {
            const response = await fetch(url, {
                method: 'GET',
                headers: { 'Accept': 'application/json' }
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.message || `HTTP ${response.status}`);
            }
            return data;
        } catch (error) {
            console.error(`[ApiService] getHistoryRecord(${id}) failed:`, error);
            throw error;
        }
    }

    /**
     * Deletes a single historical record by ID.
     * @param {number|string} id - Record ID.
     * @returns {Promise<Object>} Deletion result.
     */
    async function deleteHistoryRecord(id) {
        const url = `${BASE_URL}/api/history/${id}`;
        try {
            const response = await fetch(url, {
                method: 'DELETE',
                headers: { 'Accept': 'application/json' }
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.message || `HTTP ${response.status}`);
            }
            return data;
        } catch (error) {
            console.error(`[ApiService] deleteHistoryRecord(${id}) failed:`, error);
            throw error;
        }
    }

    /**
     * Clears all analysis history records (requires confirm).
     * @returns {Promise<Object>} Clear history result.
     */
    async function clearAllHistory() {
        const url = `${BASE_URL}/api/history?confirm=true`;
        try {
            const response = await fetch(url, {
                method: 'DELETE',
                headers: { 'Accept': 'application/json' }
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.message || `HTTP ${response.status}`);
            }
            return data;
        } catch (error) {
            console.error('[ApiService] clearAllHistory failed:', error);
            throw error;
        }
    }

    return {
        checkHealth,
        predictLeaf,
        getHistory,
        getHistoryStats,
        getHistoryRecord,
        deleteHistoryRecord,
        clearAllHistory,
        BASE_URL
    };
})();
