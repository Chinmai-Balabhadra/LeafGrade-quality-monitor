/**
 * Leaf Quality Classification and Plucking Standard Monitoring System
 * Module 7: Application Bootstrap & Main Controller
 * File: frontend/js/main.js
 */

const App = (function() {
    let toastContainer;

    function init() {
        console.log('[App] Initializing AgriLeaf AI Application...');

        // Setup toast container
        toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toastContainer';
            toastContainer.className = 'toast-container';
            document.body.appendChild(toastContainer);
        }

        // Initialize sub-controllers
        UploadController.init();
        PredictionController.init();
        if (window.HistoryController) {
            HistoryController.init();
        }

        // Setup navigation and mobile menu
        setupNavigation();

        // Check backend and model status
        verifyBackendConnection();
    }

    function setupNavigation() {
        const navToggle = document.getElementById('navToggle');
        const navLinks = document.getElementById('navLinks');

        if (navToggle && navLinks) {
            navToggle.addEventListener('click', () => {
                navLinks.classList.toggle('show');
            });

            // Close menu on link click (mobile)
            navLinks.querySelectorAll('.nav-link').forEach(link => {
                link.addEventListener('click', () => {
                    navLinks.classList.remove('show');
                });
            });
        }
    }

    async function verifyBackendConnection() {
        const dot = document.getElementById('statusDot');
        const text = document.getElementById('statusText');

        if (!dot || !text) return;

        text.textContent = 'Connecting...';
        dot.className = 'status-dot';

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

        // Animate entrance
        requestAnimationFrame(() => {
            toast.classList.add('show');
        });

        // Auto remove
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
        refreshHealth: verifyBackendConnection
    };
})();

// Bootstrap on DOM ready
document.addEventListener('DOMContentLoaded', App.init);
