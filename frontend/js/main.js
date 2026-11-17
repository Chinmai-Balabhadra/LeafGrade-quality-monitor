/**
 * Leafora – Plant Leaf Analysis & Quality Monitoring System
 * Module 7: Application Bootstrap & Main Controller
 * File: frontend/js/main.js
 */

const App = (function() {
    let toastContainer;
    const SECTIONS = ['home', 'analysis', 'history', 'model-performance', 'features', 'how-it-works', 'about'];

    function init() {
        console.log('[App] Initializing Leafora Application...');

        // Setup toast container
        toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toastContainer';
            toastContainer.className = 'toast-container';
            document.body.appendChild(toastContainer);
        }

        // Initialize sub-controllers with dual global/scope fallback and verification logging
        console.log('[App] Step 1: Initializing sub-controllers...');

        const uploadCtrl = window.UploadController || (typeof UploadController !== 'undefined' ? UploadController : null);
        if (uploadCtrl && typeof uploadCtrl.init === 'function') {
            console.log('[App] -> Initializing UploadController...');
            uploadCtrl.init();
        } else {
            console.error('[App] -> UploadController not available!');
        }

        const predCtrl = window.PredictionController || (typeof PredictionController !== 'undefined' ? PredictionController : null);
        if (predCtrl && typeof predCtrl.init === 'function') {
            console.log('[App] -> Initializing PredictionController...');
            predCtrl.init();
        } else {
            console.error('[App] -> PredictionController not available!');
        }

        const histCtrl = window.HistoryController || (typeof HistoryController !== 'undefined' ? HistoryController : null);
        if (histCtrl && typeof histCtrl.init === 'function') {
            console.log('[App] -> Initializing HistoryController...');
            histCtrl.init();
        } else {
            console.error('[App] -> HistoryController not available!');
        }

        const evalCtrl = window.ModelEvaluationController || (typeof ModelEvaluationController !== 'undefined' ? ModelEvaluationController : null);
        if (evalCtrl && typeof evalCtrl.init === 'function') {
            console.log('[App] -> Initializing ModelEvaluationController...');
            evalCtrl.init();
        } else {
            console.error('[App] -> ModelEvaluationController not available!');
        }

        // Setup navigation and section switcher
        console.log('[App] Step 2: Setting up navigation...');
        setupNavigation();

        // Load dynamic home metrics (Date, greeting, summary KPI cards)
        console.log('[App] Step 3: Loading home metrics...');
        updateHomeMetrics();

        // Check backend and model status
        console.log('[App] Step 4: Verifying backend connection...');
        verifyBackendConnection();

        console.log('[App] Leafora Application initialization complete.');
    }

    function switchSection(targetId) {
        if (!SECTIONS.includes(targetId)) return;

        // Stop active camera hardware stream when navigating away from Leaf Analysis
        if (targetId !== 'analysis' && window.UploadController && typeof UploadController.stopCamera === 'function') {
            UploadController.stopCamera();
        }

        // 1. Hide all sections, display target section
        SECTIONS.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                if (id === targetId) {
                    el.classList.add('active');
                    el.style.display = 'block';
                } else {
                    el.classList.remove('active');
                    el.style.display = 'none';
                }
            }
        });

        // 2. Update active link in sidebar and navbar
        const allLinks = document.querySelectorAll('.nav-link, .sidebar-link, a[href^="#"]');
        allLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href === `#${targetId}`) {
                link.classList.add('active');
            } else if (href && href.startsWith('#') && SECTIONS.some(s => href === `#${s}`)) {
                link.classList.remove('active');
            }
        });

        // 3. Close mobile drawer if open
        const sidebar = document.getElementById('appSidebar');
        if (sidebar) sidebar.classList.remove('open');
        const navLinks = document.getElementById('navLinks');
        if (navLinks) navLinks.classList.remove('show');

        // 4. Scroll smoothly to top
        window.scrollTo({ top: 0, behavior: 'instant' });

        // 5. Update browser URL hash without jump
        if (window.history && window.history.pushState) {
            window.history.pushState(null, null, `#${targetId}`);
        } else {
            window.location.hash = `#${targetId}`;
        }

        // 6. Section-specific refreshes
        if (targetId === 'home') {
            updateHomeMetrics();
        } else if (targetId === 'history' && window.HistoryController && typeof HistoryController.loadHistory === 'function') {
            HistoryController.loadHistory(1);
        } else if (targetId === 'model-performance' && window.ModelEvaluationController && typeof ModelEvaluationController.loadModelEvaluation === 'function') {
            ModelEvaluationController.loadModelEvaluation();
        }
    }

    function setupNavigation() {
        const navToggle = document.getElementById('navToggle');
        const appSidebar = document.getElementById('appSidebar');
        const navLinks = document.getElementById('navLinks');
        const allLinks = document.querySelectorAll('.nav-link, .sidebar-link, a[href^="#"]');

        // Mobile drawer toggle
        if (navToggle) {
            navToggle.addEventListener('click', (e) => {
                e.stopPropagation();
                if (appSidebar) appSidebar.classList.toggle('open');
                if (navLinks) navLinks.classList.toggle('show');
            });
        }

        // Close mobile drawer when clicking outside
        document.addEventListener('click', (e) => {
            if (appSidebar && appSidebar.classList.contains('open') && !appSidebar.contains(e.target) && e.target !== navToggle && !navToggle.contains(e.target)) {
                appSidebar.classList.remove('open');
                if (navLinks) navLinks.classList.remove('show');
            }
        });

        // Intercept hash navigation to switch sections
        allLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href && href.startsWith('#')) {
                const targetId = href.substring(1);
                if (SECTIONS.includes(targetId)) {
                    link.addEventListener('click', (e) => {
                        e.preventDefault();
                        switchSection(targetId);
                    });
                }
            }
        });

        // Initialize view based on current location hash or default to 'home'
        const initialHash = window.location.hash ? window.location.hash.substring(1) : 'home';
        if (initialHash === 'camera' || initialHash === 'camera-live') {
            switchSection('analysis');
            setTimeout(() => {
                if (window.UploadController && typeof UploadController.switchTab === 'function') {
                    UploadController.switchTab('camera');
                }
            }, 150);
        } else if (initialHash === 'camera-captured') {
            switchSection('analysis');
            setTimeout(async () => {
                if (window.UploadController && typeof UploadController.switchTab === 'function') {
                    await UploadController.switchTab('camera');
                    setTimeout(async () => {
                        if (typeof UploadController.captureSnapshot === 'function') {
                            await UploadController.captureSnapshot();
                        }
                    }, 1200);
                }
            }, 150);
        } else if (initialHash === 'camera-analyze') {
            switchSection('analysis');
            setTimeout(async () => {
                if (window.UploadController && typeof UploadController.switchTab === 'function') {
                    await UploadController.switchTab('camera');
                    setTimeout(async () => {
                        if (typeof UploadController.captureSnapshot === 'function') {
                            await UploadController.captureSnapshot();
                            setTimeout(() => {
                                const analyzeBtn = document.getElementById('btnAnalyzeLeaf');
                                if (analyzeBtn) analyzeBtn.click();
                            }, 500);
                        }
                    }, 1200);
                }
            }, 150);
        } else if (initialHash === 'test-browse') {
            switchSection('analysis');
            setTimeout(() => {
                console.log('[TestBrowse] Simulating leaf image file selection...');
                const canvas = document.createElement('canvas');
                canvas.width = 300;
                canvas.height = 300;
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = '#2d6a4f';
                ctx.fillRect(0, 0, 300, 300);
                ctx.fillStyle = '#52b788';
                ctx.beginPath();
                ctx.arc(150, 150, 100, 0, Math.PI * 2);
                ctx.fill();
                canvas.toBlob((blob) => {
                    const uploadCtrl = window.UploadController || UploadController;
                    if (uploadCtrl && typeof uploadCtrl.stageFile === 'function') {
                        uploadCtrl.stageFile(blob, 'apple_scab_leaf_sample.jpg');
                        console.log('[TestBrowse] Image successfully staged into previewContainer.');
                    }
                }, 'image/jpeg', 0.95);
            }, 200);
        } else if (initialHash === 'test-browse-predict') {
            switchSection('analysis');
            setTimeout(() => {
                console.log('[TestBrowsePredict] Simulating browse and analyze pipeline...');
                const canvas = document.createElement('canvas');
                canvas.width = 300;
                canvas.height = 300;
                const ctx = canvas.getContext('2d');
                ctx.fillStyle = '#2d6a4f';
                ctx.fillRect(0, 0, 300, 300);
                canvas.toBlob((blob) => {
                    const uploadCtrl = window.UploadController || UploadController;
                    if (uploadCtrl && typeof uploadCtrl.stageFile === 'function') {
                        uploadCtrl.stageFile(blob, 'apple_scab_leaf_sample.jpg');
                        setTimeout(() => {
                            const btn = document.getElementById('btnAnalyzeLeaf');
                            if (btn) btn.click();
                        }, 500);
                    }
                }, 'image/jpeg', 0.95);
            }, 200);
        } else if (initialHash === 'test-ood-face') {
            switchSection('analysis');
            setTimeout(() => {
                console.log('[TestOODFace] Simulating human face upload and OOD rejection...');
                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    canvas.width = img.naturalWidth || 600;
                    canvas.height = img.naturalHeight || 600;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    canvas.toBlob((blob) => {
                        const uploadCtrl = window.UploadController || UploadController;
                        if (uploadCtrl && typeof uploadCtrl.stageFile === 'function') {
                            uploadCtrl.stageFile(blob, 'test_human_face.jpg');
                            console.log('[TestOODFace] Face image staged. Displaying OOD rejection...');
                            setTimeout(() => {
                                const predCtrl = window.PredictionController || PredictionController;
                                if (predCtrl) {
                                    predCtrl.hideResults();
                                    predCtrl.showRejectionAlert("No valid plant leaf detected. Please upload or capture a clear photograph of a plant leaf.");
                                }
                                App.showToast("No valid plant leaf detected. Please upload or capture a clear photograph of a plant leaf.", 'error');
                            }, 500);
                        }
                    }, 'image/jpeg', 0.92);
                };
                img.src = 'assets/images/test_human_face.jpg';
            }, 300);
        } else if (initialHash === 'test-ood-accepted') {
            switchSection('analysis');
            setTimeout(() => {
                console.log('[TestOODAccepted] Simulating genuine leaf upload and acceptance...');
                const img = new Image();
                img.onload = () => {
                    const canvas = document.createElement('canvas');
                    canvas.width = img.naturalWidth || 600;
                    canvas.height = img.naturalHeight || 600;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0);
                    canvas.toBlob((blob) => {
                        const uploadCtrl = window.UploadController || UploadController;
                        if (uploadCtrl && typeof uploadCtrl.stageFile === 'function') {
                            uploadCtrl.stageFile(blob, 'valid_apple_leaf.jpg');
                            console.log('[TestOODAccepted] Genuine leaf staged. Triggering Analyze Leaf...');
                            setTimeout(() => {
                                const btn = document.getElementById('btnAnalyzeLeaf');
                                if (btn) btn.click();
                            }, 500);
                        }
                    }, 'image/jpeg', 0.92);
                };
                img.src = 'assets/images/valid_apple_leaf.jpg';
            }, 300);
        } else if (SECTIONS.includes(initialHash)) {
            switchSection(initialHash);
        } else {
            switchSection('home');
        }

        // Support browser Back and Forward buttons
        window.addEventListener('popstate', () => {
            const hash = window.location.hash ? window.location.hash.substring(1) : 'home';
            if (SECTIONS.includes(hash)) {
                switchSection(hash);
            }
        });

        // Release camera hardware on window unload
        window.addEventListener('beforeunload', () => {
            if (window.UploadController && typeof UploadController.stopCamera === 'function') {
                UploadController.stopCamera();
            }
        });
    }

    async function updateHomeMetrics() {
        // 1. Dynamic Date Display: e.g. "Tue, 17 Nov 2026"
        const dateEl = document.getElementById('currentDateDisplay');
        if (dateEl) {
            const now = new Date();
            const dateStr = now.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' });
            dateEl.textContent = dateStr;
        }

        // 2. Greeting based on local time
        const greetingEl = document.getElementById('welcomeGreeting');
        if (greetingEl) {
            const hour = new Date().getHours();
            if (hour < 12) {
                greetingEl.textContent = 'Good Morning,';
            } else if (hour < 18) {
                greetingEl.textContent = 'Good Afternoon,';
            } else {
                greetingEl.textContent = 'Good Evening,';
            }
        }

        // 3. Dynamic Summary Cards from backend
        try {
            if (typeof ApiService !== 'undefined') {
                // Fetch history stats
                const stats = await ApiService.getHistoryStats();
                if (stats && stats.status === 'SUCCESS' && stats.statistics) {
                    const s = stats.statistics;
                    const elTotal = document.getElementById('homeTotalAnalyses');
                    const elHealthy = document.getElementById('homeHealthyCount');
                    const elDiseased = document.getElementById('homeDiseasedCount');
                    if (elTotal && s.total_records !== undefined) elTotal.textContent = s.total_records;
                    if (elHealthy && s.healthy_count !== undefined) elHealthy.textContent = s.healthy_count;
                    if (elDiseased && s.diseased_count !== undefined) elDiseased.textContent = s.diseased_count;
                }

                // Fetch model evaluation accuracy
                const evalData = await ApiService.getModelEvaluation();
                if (evalData && evalData.status === 'SUCCESS' && evalData.overall_metrics) {
                    const om = evalData.overall_metrics;
                    const elAcc = document.getElementById('homeAccuracyVal');
                    if (elAcc && om.accuracy_percent !== undefined) {
                        elAcc.textContent = `${om.accuracy_percent.toFixed(2)}%`;
                    }
                }
            }
        } catch (err) {
            console.warn('[App] Could not load dynamic home metrics, using empirical baseline:', err.message);
        }
    }

    async function verifyBackendConnection() {
        const dot = document.getElementById('statusDot');
        const text = document.getElementById('statusText');

        if (!dot || !text) return;

        text.textContent = 'Connecting...';
        dot.className = 'status-dot';

        try {
            const health = await ApiService.checkHealth();
            if (health.status === 'healthy' && health.model_status && health.model_status.model_ready) {
                dot.className = 'status-dot online';
                text.textContent = 'AI Model Ready (28 Classes)';
            } else if (health.status === 'healthy') {
                dot.className = 'status-dot online';
                text.textContent = 'Backend Online';
            } else {
                dot.className = 'status-dot offline';
                text.textContent = 'Backend Offline';
                showToast('Flask backend is currently offline. Please run "python backend/app.py".', 'warning', 6000);
            }
        } catch (e) {
            dot.className = 'status-dot offline';
            text.textContent = 'Backend Offline';
        }
    }

    function showToast(message, type = 'info', duration = 4000) {
        if (!toastContainer) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type === 'error' ? 'toast-error' : (type === 'warning' ? 'toast-warning' : '')}`;

        const iconClass = type === 'error' 
            ? 'fas fa-exclamation-circle' 
            : (type === 'warning' ? 'fas fa-exclamation-triangle' : 'fas fa-check-circle');

        toast.innerHTML = `
            <i class="${iconClass}"></i>
            <span>${message}</span>
        `;

        toastContainer.appendChild(toast);

        requestAnimationFrame(() => {
            toast.classList.add('show');
        });

        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => {
                if (toast.parentElement) {
                    toast.parentElement.removeChild(toast);
                }
            }, 300);
        }, duration);
    }

    return {
        init,
        showToast,
        switchSection,
        refreshHealth: verifyBackendConnection,
        refreshHomeMetrics: updateHomeMetrics
    };
})();

// Explicitly register on global window object
window.App = App;

// Bootstrap on DOM ready
document.addEventListener('DOMContentLoaded', App.init);
