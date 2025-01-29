var links = document.getElementsByClassName("links");
var contents = document.getElementsByClassName("contents");

function isValidImage(file) {
    const validImageTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
    return validImageTypes.includes(file.type);
}

function opentab(name) {
    for (let link of links) {
        link.classList.remove("activelink");
    }
    for (let content of contents) {
        content.classList.remove("activecontent");
    }

    document.querySelector(`.links[onclick="opentab('${name}')"]`).classList.add("activelink");
    document.getElementById(name).classList.add("activecontent");
}

const wordSlider = document.querySelector('#word .slider');
const wordImages = wordSlider.querySelectorAll('img');

const characterSlider = document.querySelector('#character .slider');
const characterImages = characterSlider.querySelectorAll('img');

let currentIndex = 0;
let intervalId;

function showImage(images, index) {
    images.forEach((image, i) => {
        image.style.display = i === index ? 'block' : 'none';
    });
}

function startSlider(images) {
    intervalId = setInterval(() => {
        currentIndex = (currentIndex + 1) % images.length;
        showImage(images, currentIndex);
    }, 3000);
}

function stopSlider() {
    clearInterval(intervalId);
}

// Initialize the slider for word images
showImage(wordImages, 0);
startSlider(wordImages);

// Initialize the slider for character images
showImage(characterImages, 0);
startSlider(characterImages);

function copyText(elementId) {
    const text = document.getElementById(elementId).innerText;
    navigator.clipboard.writeText(text).then(() => {
        const button = document.querySelector(`#${elementId}`).parentElement.querySelector('.copy-btn');
        const originalText = button.innerText;
        button.innerText = 'Copied!';
        setTimeout(() => {
            button.innerText = originalText;
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text:', err);
    });
}

document.getElementById('image-upload-word').addEventListener('change', function () {
    const fileInput = this.files[0];
    if (!isValidImage(fileInput)) {
        alert('Please upload a valid image file.');
        this.value = ''; // Clear the input
        document.getElementById('file-name-word').textContent = 'No file selected!';
        return;
    }
    const fileName = fileInput?.name || 'No file selected';
    document.getElementById('file-name-word').textContent = `Selected File: ${fileName}`;

    if (fileInput) {
        const reader = new FileReader();
        reader.onload = function (e) {
            // Pause the slider and display the uploaded image
            stopSlider();

            // Dynamically replace the current image with the uploaded image
            const sliderContainer = document.querySelector('#word .slider');
            sliderContainer.innerHTML = ''; // Clear the slider content
            const imgElement = document.createElement('img');
            imgElement.src = e.target.result;
            imgElement.style.display = 'block'; // Ensure the image is visible
            sliderContainer.appendChild(imgElement);

            document.getElementById('translate-label').removeAttribute('hidden');
        };
        reader.readAsDataURL(fileInput);
    }
});

document.getElementById('image-upload-character').addEventListener('change', function () {
    const fileInput = this.files[0];
    if (!isValidImage(fileInput)) {
        alert('Please upload a valid image file.');
        this.value = ''; // Clear the input
        document.getElementById('file-name-character').textContent = 'No file selected!';
        return;
    }
    const fileName = fileInput?.name || 'No file selected';
    document.getElementById('file-name-character').textContent = `Selected File: ${fileName}`;

    if (fileInput) {
        const reader = new FileReader();
        reader.onload = function (e) {
            // Pause the slider and display the uploaded image
            stopSlider();

            // Dynamically replace the current image with the uploaded image
            const sliderContainer = document.querySelector('#character .slider');
            sliderContainer.innerHTML = ''; // Clear the slider content
            const imgElement = document.createElement('img');
            imgElement.src = e.target.result;
            imgElement.style.display = 'block'; // Ensure the image is visible
            sliderContainer.appendChild(imgElement);

            document.getElementById('translate-label').removeAttribute('hidden');
        };
        reader.readAsDataURL(fileInput);
    }
});

function showUploadButton() {
    const existingButton = document.getElementById('upload-btn');
    if (!existingButton) {
        const button = document.createElement('button');
        button.id = 'upload-btn';
        button.textContent = 'Confirm Upload';
        button.className = 'upload-btn'; // Add a class for styling
        button.addEventListener('click', () => {
            alert('Image upload confirmed!');
        });
        
        // Append the button to the card or any container
        const card = document.querySelector('.card');
        card.appendChild(button);
    }
}

function translateWord() {
    const fileInput = document.getElementById('image-upload-word').files[0];
    if (!fileInput) {
        alert('Please select an image first.');
        return;
    }

    const formData = new FormData();
    formData.append('image', fileInput);

    fetch('/perform_word_ocr/', {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': getCookie('csrftoken')
        }
    })
    .then(response => response.json())
    .then(data => {
        const outputElement = document.getElementById('word-output-text');
        outputElement.textContent = data.result;
        document.querySelector('#word-output .copy-btn').style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function translateCharacter() {
    const fileInput = document.getElementById('image-upload-character').files[0];
    if (!fileInput) {
        alert('Please select an image first.');
        return;
    }

    const formData = new FormData();
    formData.append('image', fileInput);

    fetch('/perform_char_ocr/', {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': getCookie('csrftoken')
        }
    })
    .then(response => response.json())
    .then(data => {
        const outputElement = document.getElementById('character-output-text');
        outputElement.textContent = data.result;
        document.querySelector('#character-output .copy-btn').style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

document.getElementById("signupbtn").addEventListener("click", function () {
    window.location.href = "/login/";
});