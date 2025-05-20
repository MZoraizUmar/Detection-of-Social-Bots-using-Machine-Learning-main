// 📄 script.js (Upgrade Version)

// Show spinner while detecting
function showSpinner() {
    document.getElementById('spinner').style.display = 'block';
}

// Hide spinner after prediction
function hideSpinner() {
    document.getElementById('spinner').style.display = 'none';
}

// Handle form submit to show spinner
document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('detector-form');
    if (form) {
        form.addEventListener('submit', function () {
            showSpinner();
        });
    }
});
