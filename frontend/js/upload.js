/**
 * Leaf Quality Classification and Plucking Standard Monitoring System
 * Module 7: Upload and Camera Ingestion Controller
 * File: frontend/js/upload.js
 */

const UploadController = (function() {
    // DOM Elements
    let dropzone, fileInput, browseBtn;
    let tabUpload, tabCamera, contentUpload, contentCamera;
    let cameraContainer, cameraFeed, cameraCanvas, btnCapture, btnCloseCamera, btnSwitchCamera;
    let previewContainer, previewImage, previewFilename, previewFilesize, btnRemove, btnRetake;
    
    // Internal State
    let currentFileBlob = null;
    let currentFileName = null;
    let cameraStream = null;
    let isStartingCamera = false;
    let lastStagedSource = 'upload'; // 'upload' or 'camera'

    // Detect mobile device to default to rear camera; laptops default to user (front) webcam
    const isMobileDevice = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    let currentFacingMode = isMobileDevice ? "environment" : "user";

    const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB
    const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp'];

    function init() {
        console.log('[UploadController] Initializing Upload & Camera Ingestion Controller...');
        // Cache DOM elements
        dropzone = document.getElementById('dropzone');
        fileInput = document.getElementById('leafFileInput');
        browseBtn = document.getElementById('browseFilesBtn');

        tabUpload = document.getElementById('tabUpload');
        tabCamera = document.getElementById('tabCamera');
        contentUpload = document.getElementById('contentUpload');
        contentCamera = document.getElementById('contentCamera');

        cameraContainer = document.getElementById('cameraContainer');
        cameraFeed = document.getElementById('cameraFeed');
        cameraCanvas = document.getElementById('cameraCanvas');
        btnCapture = document.getElementById('btnCapture');
        btnCloseCamera = document.getElementById('btnCloseCamera');
        btnSwitchCamera = document.getElementById('btnSwitchCamera');

        previewContainer = document.getElementById('previewContainer');
        previewImage = document.getElementById('previewImage');
        previewFilename = document.getElementById('previewFilename');
        previewFilesize = document.getElementById('previewFilesize');
        btnRemove = document.getElementById('btnRemoveImage');
        btnRetake = document.getElementById('btnRetakeImage');

        setupEventListeners();
        console.log('[UploadController] Initialized successfully. DOM elements cached and event listeners bound.');
    }

    function setupEventListeners() {
        // Tab Switching
        if (tabUpload) {
            tabUpload.addEventListener('click', () => {
                console.log('[UploadController] Upload tab clicked');
                switchTab('upload');
            });
        }
        if (tabCamera) {
            tabCamera.addEventListener('click', () => {
                console.log('[UploadController] Camera tab clicked');
                switchTab('camera');
            });
        }

        // File Input & Browse
        if (browseBtn) {
            browseBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                console.log('[UploadController] Browse button clicked -> triggering fileInput.click()');
                if (fileInput) fileInput.click();
            });
        }
        if (dropzone) {
            dropzone.addEventListener('click', (e) => {
                if (e.target === fileInput || (browseBtn && browseBtn.contains(e.target))) return;
                console.log('[UploadController] Dropzone area clicked -> triggering fileInput.click()');
                if (fileInput) fileInput.click();
            });

            // Drag and Drop
            ['dragenter', 'dragover'].forEach(eventName => {
                dropzone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropzone.classList.add('dragover');
                });
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropzone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    dropzone.classList.remove('dragover');
                });
            });

            dropzone.addEventListener('drop', (e) => {
                e.preventDefault();
                e.stopPropagation();
                console.log('[UploadController] File dropped on dropzone');
                handleFileDrop(e);
            });
        }

        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                console.log('[UploadController] File input change event triggered');
                handleFileSelect(e);
            });
        }

        // Camera Controls
        if (btnCapture) {
            btnCapture.addEventListener('click', () => {
                console.log('[UploadController] Capture Leaf button clicked');
                captureCameraSnapshot();
            });
        }
        if (btnCloseCamera) {
            btnCloseCamera.addEventListener('click', () => {
                console.log('[UploadController] Close Camera button clicked');
                switchTab('upload');
            });
        }
        if (btnSwitchCamera) {
            btnSwitchCamera.addEventListener('click', () => {
                console.log('[UploadController] Switch/Flip Camera button clicked');
                toggleCameraFacing();
            });
        }

        // Preview Actions
        if (btnRemove) {
            btnRemove.addEventListener('click', () => {
                console.log('[UploadController] Remove Image button clicked');
                resetSelection(true);
            });
        }
        if (btnRetake) {
            btnRetake.addEventListener('click', () => {
                console.log('[UploadController] Retake Image button clicked');
                resetSelection(false);
            });
        }
    }

    async function switchTab(tab) {
        // Ensure staged preview is dismissed when switching tabs
        if (previewContainer) {
            previewContainer.classList.remove('active');
        }

        if (tab === 'upload') {
            if (tabUpload) tabUpload.classList.add('active');
            if (tabCamera) tabCamera.classList.remove('active');
            if (contentUpload) contentUpload.classList.add('active');
            if (contentCamera) contentCamera.classList.remove('active');
            stopCameraStream();
            return null;
        } else if (tab === 'camera') {
            if (tabCamera) tabCamera.classList.add('active');
            if (tabUpload) tabUpload.classList.remove('active');
            if (contentCamera) contentCamera.classList.add('active');
            if (contentUpload) contentUpload.classList.remove('active');
            return await startCameraStream();
        }
    }

    function validateFile(file) {
        if (!file) return false;

        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!ALLOWED_EXTENSIONS.includes(ext)) {
            App.showToast(`Unsupported file format (${ext}). Please upload JPG, PNG, BMP, or WEBP.`, 'error');
            return false;
        }

        if (file.size > MAX_FILE_SIZE) {
            App.showToast(`File exceeds 16 MB limit (${(file.size / (1024*1024)).toFixed(1)} MB).`, 'error');
            return false;
        }

        return true;
    }

    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file && validateFile(file)) {
            lastStagedSource = 'upload';
            stageFile(file, file.name);
        }
    }

    function handleFileDrop(e) {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        if (file && validateFile(file)) {
            lastStagedSource = 'upload';
            stageFile(file, file.name);
        }
    }

    function stageFile(blobOrFile, filename) {
        currentFileBlob = blobOrFile;
        currentFileName = filename || 'leaf_sample.jpg';

        const objectUrl = URL.createObjectURL(blobOrFile);
        if (previewImage) previewImage.src = objectUrl;
        if (previewFilename) previewFilename.textContent = currentFileName;
        
        const sizeKb = (blobOrFile.size / 1024).toFixed(1);
        if (previewFilesize) {
            previewFilesize.textContent = sizeKb > 1024 ? `${(sizeKb/1024).toFixed(2)} MB` : `${sizeKb} KB`;
        }

        // Switch to preview mode
        if (contentUpload) contentUpload.classList.remove('active');
        if (contentCamera) contentCamera.classList.remove('active');
        if (previewContainer) previewContainer.classList.add('active');

        // Dismiss previous rejection notice if any
        if (window.PredictionController && typeof PredictionController.hideRejectionAlert === 'function') {
            PredictionController.hideRejectionAlert();
        }

        // Immediately stop camera tracks to release webcam hardware
        stopCameraStream();

        App.showToast('Leaf image staged for analysis.', 'info');
    }

    // Camera APIs with robust fallback and error diagnostics
    async function startCameraStream() {
        if (isStartingCamera) {
            console.warn('[UploadController] Camera initialization already in progress.');
            return;
        }

        // Verify MediaDevices API availability
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            App.showToast('Camera API is not supported in this browser or requires a secure context (http://localhost or https).', 'warning', 6000);
            switchTab('upload');
            return;
        }

        isStartingCamera = true;
        stopCameraStream(); // Terminate any lingering stream

        let stream = null;

        try {
            // Attempt 1: Ideal facingMode and resolution (works for mobile & advanced webcams)
            try {
                const primaryConstraints = {
                    video: {
                        facingMode: { ideal: currentFacingMode },
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    },
                    audio: false
                };
                stream = await navigator.mediaDevices.getUserMedia(primaryConstraints);
            } catch (primaryErr) {
                console.warn('[UploadController] Ideal camera constraints unavailable, attempting fallback:', primaryErr.name, primaryErr.message);
                
                // Attempt 2: Universal unconstrained video (guaranteed for laptop/USB webcams)
                stream = await navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: false
                });
            }

            if (!stream) {
                throw new Error('Failed to acquire video stream from camera hardware.');
            }

            cameraStream = stream;

            if (cameraFeed) {
                cameraFeed.srcObject = cameraStream;

                // Ensure video begins playback smoothly
                try {
                    await cameraFeed.play();
                } catch (playErr) {
                    console.warn('[UploadController] Direct play interrupted, awaiting metadata:', playErr);
                    await new Promise((resolve) => {
                        cameraFeed.onloadedmetadata = () => {
                            cameraFeed.play().then(resolve).catch(resolve);
                        };
                        setTimeout(resolve, 1500); // Safety fallback timeout
                    });
                }
            }

            App.showToast('Live camera feed active.', 'info');

        } catch (err) {
            handleCameraError(err);
        } finally {
            isStartingCamera = false;
        }
    }

    function handleCameraError(err) {
        console.error('[UploadController] Camera access failure:', err);
        let userMessage = 'Unable to access camera hardware.';

        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            userMessage = 'Camera permission was denied. Please allow camera access in your browser address bar to use this feature.';
        } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
            userMessage = 'No camera device detected on your system. Please connect a webcam or use file upload.';
        } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
            userMessage = 'Camera is in use by another application. Please close other camera apps and retry.';
        } else if (err.name === 'OverconstrainedError') {
            userMessage = 'Camera does not meet requested resolution criteria. Please try again.';
        } else if (err.name === 'SecurityError') {
            userMessage = 'Camera access requires a secure origin (http://127.0.0.1 or https).';
        } else if (window.location.protocol === 'file:') {
            userMessage = 'Camera capture is restricted under file:// protocol. Please open the app via http://127.0.0.1:5000.';
        } else if (err.message) {
            userMessage = `Camera error: ${err.message}`;
        }

        App.showToast(userMessage, 'error', 6000);
        switchTab('upload');
    }

    function stopCameraStream() {
        if (cameraStream) {
            try {
                cameraStream.getTracks().forEach(track => {
                    track.stop();
                });
            } catch (e) {
                console.warn('[UploadController] Error stopping media tracks:', e);
            }
            cameraStream = null;
        }
        if (cameraFeed) {
            cameraFeed.srcObject = null;
        }
    }

    async function toggleCameraFacing() {
        currentFacingMode = currentFacingMode === "user" ? "environment" : "user";
        App.showToast(`Switching camera to ${currentFacingMode === 'user' ? 'front' : 'rear'}...`, 'info', 2000);
        await startCameraStream();
    }

    async function captureCameraSnapshot() {
        if (!cameraFeed || !cameraStream) {
            App.showToast('Camera feed is not active. Please start camera first.', 'warning');
            return;
        }

        // If video dimensions are not yet populated, give it a moment
        if ((cameraFeed.videoWidth || 0) === 0) {
            await new Promise((resolve) => {
                cameraFeed.onloadedmetadata = () => resolve();
                setTimeout(resolve, 800);
            });
        }

        const videoWidth = cameraFeed.videoWidth || cameraFeed.clientWidth || 640;
        const videoHeight = cameraFeed.videoHeight || cameraFeed.clientHeight || 480;

        if (!cameraCanvas) {
            cameraCanvas = document.createElement('canvas');
        }

        cameraCanvas.width = videoWidth;
        cameraCanvas.height = videoHeight;
        const ctx = cameraCanvas.getContext('2d');
        ctx.drawImage(cameraFeed, 0, 0, videoWidth, videoHeight);

        function processBlob(blob) {
            if (blob && blob.size > 0) {
                lastStagedSource = 'camera';
                const now = new Date();
                const timestamp = now.toISOString().replace(/[:.]/g, '-');
                stageFile(blob, `leaf_capture_${timestamp}.jpg`);
            } else {
                App.showToast('Failed to generate image from camera feed.', 'error');
            }
        }

        if (cameraCanvas.toBlob) {
            cameraCanvas.toBlob(processBlob, 'image/jpeg', 0.95);
        } else {
            // Fallback for environments where toBlob is absent
            try {
                const dataUrl = cameraCanvas.toDataURL('image/jpeg', 0.95);
                fetch(dataUrl)
                    .then(res => res.blob())
                    .then(processBlob)
                    .catch(() => App.showToast('Failed to convert canvas to image.', 'error'));
            } catch (canvasErr) {
                App.showToast('Canvas capture error: ' + canvasErr.message, 'error');
            }
        }
    }

    function resetSelection(forceUpload = false) {
        currentFileBlob = null;
        currentFileName = null;
        if (fileInput) fileInput.value = '';
        if (previewImage) previewImage.src = '';

        if (previewContainer) {
            previewContainer.classList.remove('active');
        }

        // Dismiss previous rejection notice if any
        if (window.PredictionController && typeof PredictionController.hideRejectionAlert === 'function') {
            PredictionController.hideRejectionAlert();
        }

        // If retaking and previous input was camera, reopen camera; otherwise return to upload
        if (!forceUpload && lastStagedSource === 'camera') {
            switchTab('camera');
        } else {
            switchTab('upload');
        }

        // Hide previous diagnosis results when resetting
        if (window.PredictionController && typeof PredictionController.hideResults === 'function') {
            PredictionController.hideResults();
        }
    }

    return {
        init,
        getCurrentBlob: () => currentFileBlob,
        getCurrentFilename: () => currentFileName,
        resetSelection,
        switchTab,
        stageFile,
        startCamera: startCameraStream,
        stopCamera: stopCameraStream,
        captureSnapshot: captureCameraSnapshot
    };
})();

// Explicitly register on global window object for resilient cross-script lifecycle
window.UploadController = UploadController;
