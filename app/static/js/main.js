// app/static/js/main.js
let currentFile = null;

document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        alert(result.message);
        
        if (response.ok) {
            currentFile = file.name;
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error uploading file');
    }
});

// app/static/js/main.js
document.getElementById('chat-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const messageInput = document.getElementById('message-input');
    const imageInput = document.getElementById('image-input');
    const message = messageInput.value.trim();
    
    if (!message) return;

    // Add user message to chat
    addMessage(message, 'user');
    
    // If there's an image, display it in chat
    if (imageInput.files[0]) {
        addImage(URL.createObjectURL(imageInput.files[0]), 'user');
    }
    
    messageInput.value = '';
    imageInput.value = '';

    const formData = new FormData();
    formData.append('message', message);
    if (currentFile) {
        formData.append('file_name', currentFile);
    }
    if (imageInput.files[0]) {
        formData.append('image', imageInput.files[0]);
    }

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        addMessage(result.response, 'bot');
    } catch (error) {
        console.error('Error:', error);
        addMessage('Error: Could not get response', 'bot');
    }
});

function addImage(imageUrl, sender) {
    const chatMessages = document.getElementById('chat-messages');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', `${sender}-message`);
    
    const img = document.createElement('img');
    img.src = imageUrl;
    img.classList.add('chat-image');
    
    messageElement.appendChild(img);
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addMessage(message, sender) {
    const chatMessages = document.getElementById('chat-messages');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', `${sender}-message`);
    messageElement.textContent = message;
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}