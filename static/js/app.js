// DOM Elements
const uploadContainer = document.getElementById('uploadContainer');
const fileInput = document.getElementById('fileInput');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const uploadLoading = document.getElementById('uploadLoading');
const uploadBtn = document.getElementById('uploadBtn');
const changeImageBtn = document.getElementById('changeImageBtn');
const generateBtn = document.getElementById('generateBtn');
const themeToggle = document.getElementById('themeToggle');

const sidebar = document.getElementById('sidebar');
const sidebarEmpty = document.getElementById('sidebarEmpty');
const captionResult = document.getElementById('captionResult');
const captionText = document.getElementById('captionText');
const inferenceTime = document.getElementById('inferenceTime');
const generationMethod = document.getElementById('generationMethod');
const descriptionText = document.getElementById('descriptionText');
const sceneTags = document.getElementById('sceneTags');
const copyCaption = document.getElementById('copyCaption');
const shareResult = document.getElementById('shareResult');

const toast = document.getElementById('toast');
const toastIcon = document.getElementById('toastIcon');
const toastMessage = document.getElementById('toastMessage');

const recentSection = document.getElementById('recentSection');
const recentGrid = document.getElementById('recentGrid');

// State
let currentFile = null;
let recentCaptions = JSON.parse(localStorage.getItem('recentCaptions') || '[]');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadTheme();
    renderRecentCaptions();
    setupEventListeners();
});

function setupEventListeners() {
    // Upload button
    uploadBtn.addEventListener('click', () => fileInput.click());
    changeImageBtn.addEventListener('click', () => fileInput.click());
    
    // File input
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadContainer.addEventListener('dragover', handleDragOver);
    uploadContainer.addEventListener('dragleave', handleDragLeave);
    uploadContainer.addEventListener('drop', handleDrop);
    
    // Generate caption
    generateBtn.addEventListener('click', generateCaption);
    
    // Theme toggle
    themeToggle.addEventListener('click', toggleTheme);
    
    // Copy caption
    copyCaption.addEventListener('click', () => {
        const text = captionText.textContent;
        navigator.clipboard.writeText(text).then(() => {
            showToast('Caption copied to clipboard!', 'check_circle');
        });
    });
    
    // Share result
    shareResult.addEventListener('click', shareCaption);
}

// File Handling
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadContainer.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadContainer.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadContainer.classList.remove('dragover');
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        processFile(file);
    } else {
        showToast('Please upload an image file', 'error');
    }
}

function processFile(file) {
    currentFile = file;
    const reader = new FileReader();
    
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadPlaceholder.style.display = 'none';
        imagePreview.style.display = 'block';
        uploadLoading.style.display = 'none';
    };
    
    reader.readAsDataURL(file);
}

// Caption Generation
async function generateCaption() {
    if (!currentFile) {
        showToast('Please upload an image first', 'warning');
        return;
    }
    
    // Show loading
    uploadLoading.style.display = 'flex';
    generateBtn.disabled = true;
    
    try {
        const formData = new FormData();
        formData.append('file', currentFile);
        
        const response = await fetch('/caption', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to generate caption');
        }
        
        const result = await response.json();
        displayCaption(result);
        saveToRecent(result);
        showToast('Caption generated successfully!', 'check_circle');
        
    } catch (error) {
        showToast(error.message, 'error');
        console.error('Error:', error);
    } finally {
        uploadLoading.style.display = 'none';
        generateBtn.disabled = false;
    }
}

function displayCaption(result) {
    // Update sidebar content
    captionText.textContent = result.caption;
    inferenceTime.textContent = `${result.inference_time_ms}ms`;
    generationMethod.textContent = result.generation_method || 'CNN+LSTM';
    
    // Generate description
    const description = generateDescription(result);
    descriptionText.textContent = description;
    
    // Generate scene tags
    const tags = generateSceneTags(result);
    sceneTags.innerHTML = tags.map(tag => `<span class="tag">${tag}</span>`).join('');
    
    // Show caption result, hide empty state
    sidebarEmpty.style.display = 'none';
    captionResult.style.display = 'flex';
    
    // Open sidebar on mobile
    if (window.innerWidth <= 1024) {
        sidebar.classList.add('open');
    }
}

function generateDescription(result) {
    const caption = result.caption;
    const method = result.generation_method || '';
    
    let description = `This image shows "${caption}". `;
    
    if (method.includes('outdoor')) {
        description += 'The scene appears to be taken outdoors with natural or urban elements visible. ';
    } else if (method.includes('indoor')) {
        description += 'The scene appears to be an indoor environment with artificial lighting and furnishings. ';
    } else if (method.includes('nature')) {
        description += 'The scene features natural elements like vegetation, landscape, or outdoor scenery. ';
    }
    
    description += `The caption was generated using a CNN (ResNet-50) for feature extraction and LSTM for text generation, with an inference time of ${result.inference_time_ms}ms.`;
    
    return description;
}

function generateSceneTags(result) {
    const method = result.generation_method || '';
    const tags = ['CNN+LSTM'];
    
    if (method.includes('outdoor')) tags.push('Outdoor');
    if (method.includes('indoor')) tags.push('Indoor');
    if (method.includes('nature')) tags.push('Nature');
    if (method.includes('person')) tags.push('People');
    
    if (result.inference_time_ms < 50) {
        tags.push('Fast Inference');
    }
    
    return tags;
}

// Recent Captions
function saveToRecent(result) {
    const item = {
        image: previewImage.src,
        caption: result.caption,
        timestamp: Date.now()
    };
    
    recentCaptions.unshift(item);
    if (recentCaptions.length > 6) {
        recentCaptions = recentCaptions.slice(0, 6);
    }
    
    localStorage.setItem('recentCaptions', JSON.stringify(recentCaptions));
    renderRecentCaptions();
}

function renderRecentCaptions() {
    if (recentCaptions.length === 0) {
        recentSection.style.display = 'none';
        return;
    }
    
    recentSection.style.display = 'block';
    recentGrid.innerHTML = recentCaptions.map((item, index) => `
        <div class="recent-item" onclick="loadRecentItem(${index})">
            <img src="${item.image}" alt="Recent">
            <p>${item.caption}</p>
        </div>
    `).join('');
}

function loadRecentItem(index) {
    const item = recentCaptions[index];
    if (item) {
        previewImage.src = item.image;
        captionText.textContent = item.caption;
        sidebarEmpty.style.display = 'none';
        captionResult.style.display = 'flex';
        
        if (window.innerWidth <= 1024) {
            sidebar.classList.add('open');
        }
    }
}

// Theme
function loadTheme() {
    const theme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', theme);
    updateThemeIcon(theme);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    updateThemeIcon(next);
}

function updateThemeIcon(theme) {
    const icon = themeToggle.querySelector('.material-icons');
    icon.textContent = theme === 'dark' ? 'light_mode' : 'dark_mode';
}

// Share
async function shareCaption() {
    const text = captionText.textContent;
    
    if (navigator.share) {
        try {
            await navigator.share({
                title: 'AI Vision Caption',
                text: text
            });
        } catch (error) {
            if (error.name !== 'AbortError') {
                showToast('Failed to share', 'error');
            }
        }
    } else {
        navigator.clipboard.writeText(text).then(() => {
            showToast('Caption copied for sharing!', 'check_circle');
        });
    }
}

// Toast Notification
function showToast(message, icon = 'check_circle') {
    toastMessage.textContent = message;
    toastIcon.textContent = icon;
    
    if (icon === 'error') {
        toastIcon.style.color = 'var(--error)';
    } else if (icon === 'warning') {
        toastIcon.style.color = 'var(--warning)';
    } else {
        toastIcon.style.color = 'var(--success)';
    }
    
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Close sidebar on outside click (mobile)
document.addEventListener('click', (e) => {
    if (window.innerWidth <= 1024 && 
        !sidebar.contains(e.target) && 
        !e.target.closest('.mobile-sidebar-toggle')) {
        sidebar.classList.remove('open');
    }
});
