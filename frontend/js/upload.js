/**
 * Leaf Quality Classification and Plucking Standard Monitoring System
 * Module 7: Upload and Camera Ingestion Controller
 * File: frontend/js/upload.js
 */

const UploadController = (function() {
    // DOM Elements
    let dropzone, fileInput, browseBtn;
    let tabUpload, tabCamera, contentUpload, contentCamera;
    let cameraContainer, cameraFeed, cameraCanvas, btnCapture, btnRetakeCamera, btnCloseCamera, btnSwitchCamera;
    let previewContainer, previewImage, previewFilename, previewFilesize, btnRemove, btnRetake;
    
    // Internal State
    let currentFileBlob = null;
    let currentFileName = null;
    let cameraStream = null;
    let currentFacingMode = "environment"; // Prefer rear camera on mobile
    const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB
    const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp'];

    function init() {
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
    }

    function setupEventListeners() {
        // Tab Switching
        tabUpload.addEventListener('click', () => switchTab('upload'));
        tabCamera.addEventListener('click', () => switchTab('camera'));

        // File Input & Browse
        browseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
        dropzone.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', handleFileSelect);

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

        dropzone.addEventListener('drop', handleFileDrop);

        // Camera Controls
        btnCapture.addEventListener('click', captureCameraSnapshot);
        btnCloseCamera.addEventListener('click', stopCameraStream);
        if (btnSwitchCamera) {
            btnSwitchCamera.addEventListener('click', toggleCameraFacing);
        }

        // Preview Actions
        btnRemove.addEventListener('click', resetSelection);
        btnRetake.addEventListener('click', resetSelection);
    }

    function switchTab(tab) {
        if (tab === 'upload') {
            tabUpload.classList.add('active');
            tabCamera.classList.remove('active');
            contentUpload.classList.add('active');
            contentCamera.classList.remove('active');
            stopCameraStream();
        } else if (tab === 'camera') {
            tabCamera.classList.add('active');
            tabUpload.classList.remove('active');
            contentCamera.classList.add('active');
            contentUpload.classList.remove('active');
            startCameraStream();
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
            stageFile(file, file.name);
        }
    }

    function handleFileDrop(e) {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        if (file && validateFile(file)) {
            stageFile(file, file.name);
        }
    }

    function stageFile(blobOrFile, filename) {
        currentFileBlob = blobOrFile;
        currentFileName = filename || 'leaf_sample.jpg';

        const objectUrl = URL.createObjectURL(blobOrFile);
        previewImage.src = objectUrl;
        previewFilename.textContent = currentFileName;
        
        const sizeKb = (blobOrFile.size / 1024).toFixed(1);
        previewFilesize.textContent = sizeKb > 1024 ? `${(sizeKb/1024).toFixed(2)} MB` : `${sizeKb} KB`;

        // Switch to preview mode
        contentUpload.classList.remove('active');
        contentCamera.classList.remove('active');
        previewContainer.classList.add('active');

        // Stop camera if running
        stopCameraStream();

        App.showToast('Leaf image staged for analysis.', 'info');
    }

    // Camera APIs
    async function startCameraStream() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            App.showToast('Camera API is not supported in this browser. Please use file upload.', 'warning');
            switchTab('upload');
            return;
        }

        try {
            stopCameraStream(); // Stop any active stream first
            const constraints = {
                video: {
                    facingMode: currentFacingMode,
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            };
            cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
            cameraFeed.srcObject = cameraStream;
            cameraFeed.play();
        } catch (err) {
            console.error('Camera access error:', err);
            App.showToast('Unable to access camera. Check device permissions or use file upload.', 'error');
            switchTab('upload');
        }
    }

    function stopCameraStream() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        if (cameraFeed) {
            cameraFeed.srcObject = null;
        }
    }

    function toggleCameraFacing() {
        currentFacingMode = currentFacingMode === "environment" ? "user" : "environment";
        startCameraStream();
    }

    function captureCameraSnapshot() {
        if (!cameraFeed || !cameraStream) return;

        const videoWidth = cameraFeed.videoWidth || 640;
        const videoHeight = cameraFeed.videoHeight || 480;

        cameraCanvas.width = videoWidth;
        cameraCanvas.height = videoHeight;
        const ctx = cameraCanvas.getContext('2d');
        ctx.drawImage(cameraFeed, 0, 0, videoWidth, videoHeight);

        cameraCanvas.toBlob((blob) => {
            if (blob) {
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                stageFile(blob, `camera_capture_${timestamp}.jpg`);
            } else {
                App.showToast('Failed to generate image from camera feed.', 'error');
            }
        }, 'image/jpeg', 0.92);
    }

    function resetSelection() {
        currentFileBlob = null;
        currentFileName = null;
        fileInput.value = '';
        previewImage.src = '';

        previewContainer.classList.remove('active');
        switchTab('upload');

        // Hide results when resetting
        PredictionController.hideResults();
    }

    return {
        init,
        getCurrentBlob: () => currentFileBlob,
        getCurrentFilename: () => currentFileName,
        resetSelection,
        stopCamera: stopCameraStream
    };
})();
