const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('fileElem');
const preview = document.getElementById('preview');
const submitButton = document.getElementById('submit-btn');
const resetButton = document.getElementById('reset-btn');
const form = document.getElementById('upload-form');
const responseMessage = document.getElementById('response-message');
const output = document.getElementById('output');
const jokesMessage = document.getElementById('jokes-message');
const positive_percentage = document.getElementById('positive-percentage');
const negative_percentage = document.getElementById('negative-percentage');
const video = document.getElementById('video');
const cameraSelect = document.getElementById('camera-select');
const detectCamerasBtn = document.getElementById('detect-cameras-btn');
const submitRecordBtn = document.getElementById('capture-btn');
const automaticRecordBtn = document.getElementById('automatic-record-btn');
const stopCameraBtn = document.getElementById('stop-camera-btn');
const outputWrapper = document.getElementById('output-wrapper');
submitRecordBtn.disabled = true;
automaticRecordBtn.disabled = true;
stopCameraBtn.disabled = true;

setTimeout(() => {
    const titleContainer = document.querySelector('.title-container');
    if (titleContainer) {
        titleContainer.style.display = 'none';
    }
}, 7000);

if (!dropArea) {
  console.error('Drop area not found');
}
if (!fileInput) {
    console.error('File input not found');
}
if (!preview) {
    console.error('Preview area not found');
}
if (!resetButton) {
    console.error('Reset button not found');
}
if (!form) {
    console.error('Form not found');
}
if (!responseMessage) {
    console.error('Response message area not found');
}
let videoDevices = [];
async function getCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        videoDevices = devices.filter(device => device.kind === 'videoinput');
        const currentDeviceId = cameraSelect.value;
        cameraSelect.innerHTML = '';

        videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || `Camera ${index + 1}`;
            cameraSelect.appendChild(option);
        });

        if (videoDevices.length > 0) {
            submitRecordBtn.disabled = false;
            automaticRecordBtn.disabled = false;
            detectCamerasBtn.disabled = true;

        } else {
            submitRecordBtn.disabled = true;
            automaticRecordBtn.disabled = true;
            stopCameraBtn.disabled = true;
            detectCamerasBtn.disabled = false;
        }

        // Si la cámara actual sigue disponible, no la reinicies
        const stillAvailable = videoDevices.some(d => d.deviceId === currentDeviceId);
        if (stillAvailable) {
            cameraSelect.value = currentDeviceId;
        } else if (videoDevices.length > 0) {
            startCamera(videoDevices[0].deviceId);
        }
    } catch (err) {
        console.error("Error listing cameras:", err);
        alert("Failed to list cameras. Make sure your device allows camera access.");
    }
}


let stream = null;
async function startCamera(deviceId = null) {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }

  const constraints = {
    video: deviceId ? { deviceId: { exact: deviceId } } : true,
    audio: false
  };

  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    submitRecordBtn.disabled = false;
  } catch (err) {
    console.error("Error accessing camera:", err.name);

    if (err.name === "NotReadableError") {
        alert("The camera is being used by another application. Please close it and try again.");
    } else if(err.name === "OverconstrainedError") {
      alert("The selected camera is not available or not supported. Please select another one.");
    } else if (err.name === "NotAllowedError") {
      alert("Access to the camera was denied.");
    } else if (err.name === "NotFoundError") {
      alert("No camera was found available.");
    } else {
      alert("Error accessing the camera: " + err.message);
    }
  }
}

cameraSelect.addEventListener('change', () => {
    startCamera(cameraSelect.value);
});

function showImage(file) {
  const reader = new FileReader();
  reader.onload = function(e) {
    preview.innerHTML = `<img src="${e.target.result}" alt="Image loaded" />`;
  };
  reader.readAsDataURL(file);
}
fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        showImage(fileInput.files[0]);
    }
    });
dropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropArea.classList.add('dragover');
});

dropArea.addEventListener('dragleave', () => {
    dropArea.classList.remove('dragover');
});

dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dropArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        showImage(e.dataTransfer.files[0]);
        fileInput.files = e.dataTransfer.files;
    }
});

resetButton.addEventListener('click', () => {
    reset_function();
});

function reset_function() {
    fileInput.value = '';
    responseMessage.textContent = '';
    output.textContent = '';
    jokesMessage.textContent = '';
    preview.innerHTML = '<p>No image loaded</p>';
    const btn = document.getElementById('submit-changes');
    if (btn) btn.remove();
    submitButton.disabled = false;
    positive_percentage.textContent = '';
    negative_percentage.textContent = '';
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        video.srcObject = null;
        videoDevices = [];
        cameraSelect.innerHTML = '';
        submitRecordBtn.disabled = true;
        stopCameraBtn.disabled = true;
        automaticRecordBtn.disabled = true;
    }
    detectCamerasBtn.disabled = false;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (fileInput.files.length === 0) {
      responseMessage.textContent = 'No image selected';
      return;
  }

  const formData = new FormData();
  formData.append('image', fileInput.files[0]);
  try {
    responseMessage.textContent = 'Processing image...';
    disableButtons();
    post_to_server(formData, "normal");

  } catch (err) {
      responseMessage.textContent = 'Error sending image';
      console.error(err);
      enableButtons();
  }
});

submitRecordBtn.addEventListener('click', async () => {
    if (stream) {
        const track = stream.getVideoTracks()[0];
        const imageCapture = new ImageCapture(track);
        disableButtons();
        try {
            responseMessage.textContent = 'Processing image from camera...';
            const blob = await imageCapture.takePhoto();
            const file = new File([blob], 'captured_image.jpg', { type: 'image/jpeg' });
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
            const formData = new FormData();
            formData.append('image', file);
            
            post_to_server(formData , "normal");
        } catch (err) {
            console.error("Error capturing photo:", err);
            console.error(err);
            enableButtons();
        }
    } else {
        alert("No camera stream available.");
    }
});

let captureInterval = null;
automaticRecordBtn.addEventListener('click', async () => {
    if (stream) {
        stopCameraBtn.disabled = false;
        responseMessage.textContent = "Automatic camera recording...";

        if (captureInterval !== null) {
            clearInterval(captureInterval);
        }

        const takeAndSendPhoto = async () => {
            const track = stream.getVideoTracks()[0];
            const imageCapture = new ImageCapture(track);
            try {
                disableButtons();
                const blob = await imageCapture.takePhoto();
                const file = new File([blob], 'captured_image.jpg', { type: 'image/jpeg' });
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                fileInput.files = dataTransfer.files;
                const formData = new FormData();
                formData.append('image', file);
                if (captureInterval === null) {
                    post_to_server(formData, "automatic");
                } else {
                    post_to_server(formData, "normal");
                }
                responseMessage.textContent = "Automatic camera recording...";
            } catch (err) {
                console.error("Error capturing photo:", err);
            }
        };

        await takeAndSendPhoto();

        captureInterval = setInterval(takeAndSendPhoto, 15000);
    }
});

stopCameraBtn.addEventListener('click', () => {
    responseMessage.innerHTML = 'Camera stopped.';
    enableButtons();
    stopCameraBtn.disabled = true;
    if (captureInterval !== null) {
        clearInterval(captureInterval);
        captureInterval = null;
        responseMessage.textContent = "Image capturing stopped. Camera still on. If takes too long press reset button or refresh the page.";
        disableButtons();
    }
});



detectCamerasBtn.addEventListener('click', () => {
    getCameras();
});

async function post_to_server(sending_data, type) {
    try {
        const record_res = await fetch('/process', {
            method: 'POST',
            body: sending_data,
        });

        if (!record_res.ok) {
            throw new Error(`Server returned ${record_res.status}`);
        }

        const data = await record_res.json();

        if (Array.isArray(data)) {
            console.log("Respuesta backend:", data);
            output.innerHTML = data.join("");
            if (type === "automatic") {
                jokesMessage.textContent = '';
                responseMessage.textContent = 'Image captured and processed successfully!';
                response_function(data);
            }else {
                response_function(data);
                jokesMessage.textContent = '';
                responseMessage.textContent = 'Image processed successfully!';
                enableButtons();
            }

        } else {
            responseMessage.textContent = data.error || 'Unexpected format';
            enableButtons();
        }

    } catch (err) {
        console.error("Error en post_to_server:", err);
        responseMessage.textContent = 'Error processing the image';
        enableButtons();
        stopCameraBtn.disabled = true;
    }
}

show_percentages = (emotions, puntuation_emotions) => {
    let totalPositive = 0;
    let totalNegative = 0;

    document.querySelectorAll('.card').forEach(card => {
        const emotion = card.dataset.emotion;
        const index = emotions.indexOf(emotion);
        const score = puntuation_emotions[index];

        if (score > 0) {
            totalPositive += score;
        } else if (score < 0) {
            totalNegative += Math.abs(score);
        }
    });

    // Calcular porcentaje total
    const total = totalPositive + totalNegative;
    let positivePct = 0;
    let negativePct = 0;

    if (total > 0) {
        positivePct = Math.round((totalPositive / total) * 100);
        negativePct = 100 - positivePct;
    }

    positive_percentage.textContent = `Positive: ${positivePct}%`;
    negative_percentage.textContent = `Negative: ${negativePct}%`;
}

post_emotions = async () => {
    if (!document.getElementById('submit-changes')) {
        const submitChangesBtn = document.createElement('button');
        submitChangesBtn.id = 'submit-changes';
        submitChangesBtn.textContent = 'Submit Emotion Changes';
        submitChangesBtn.classList.add('btn');
        submitChangesBtn.style.marginTop = '1rem';
    
        outputWrapper.appendChild(submitChangesBtn);
    
        // Asignar comportamiento
        submitChangesBtn.addEventListener('click', async () => {
            const changes = [];
    
            document.querySelectorAll('.card').forEach(card => {
                const name = card.dataset.name;
                const originalEmotion = card.dataset.emotion;
                const selectedEmotion = card.querySelector('select').value;
                const image = card.querySelector('img').src;
    
                if (originalEmotion !== selectedEmotion) {
                    changes.push({ name, new_emotion: selectedEmotion, image_path: image });
                }
            });
    
            if (changes.length === 0) {
                alert("No changes detected.");
                return;
            }
    
            try {
                const res = await fetch('/update_emotions', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(changes),
                });
    
                const result = await res.json();
    
                if (res.ok) {
                    alert("Emotions updated successfully.");
                } else {
                    alert(result.error || "Failed to update emotions.");
                }
            } catch (err) {
                console.error("Error submitting changes:", err);
                alert("An error occurred while submitting changes.");
            }
        });
    }
}

const response_function = async (data) => {
    const emotions = ['happy', 'surprise', 'neutral', 'disgust', 'sad', 'fear',  'angry'];
    const puntuation_emotions = [5, 3, 1, -1, -2, -3, -5]

    document.querySelectorAll('.card').forEach(card => {
        const dropdown = document.createElement('select');
        emotions.forEach(emotion => {
            const option = document.createElement('option');
            option.value = emotion;
            option.textContent = emotion;
            if (card.dataset.emotion === emotion) {
                option.selected = true;
            }
            dropdown.appendChild(option);
        });
        card.appendChild(dropdown);
    });
    show_percentages(emotions, puntuation_emotions);
    post_emotions();

    try{
    if (data.some(emotion => emotion.includes("disgust") || emotion.includes("sad"))) {
        const selectedLengauge = document.querySelector('input[name="lenguage"]:checked');

        const joke_res = await fetch('/process_jokes', {
            method: 'POST',
            headers: {
                'Content-Type': 'text/plain',
            },
            body: selectedLengauge.value,
        });
        const jokes = await joke_res.json();
        if (joke_res.ok) {
            console.log(jokes);
            jokesMessage.textContent = jokes.joke.join("\n");
        } else {
            jokesMessage.textContent = jokes.error || 'Error fetching jokes';
        }
        }
    } catch (err) {
    console.error('Error fetching jokes:', err);
    jokesMessage.textContent = 'Error fetching jokes';
    }
}

function disableButtons() {
    submitButton.disabled = true;
    submitRecordBtn.disabled = true;
    automaticRecordBtn.disabled = true;
    detectCamerasBtn.disabled = true;
}

function enableButtons() {
    submitButton.disabled = false;
    if (videoDevices.length > 0) {
        submitRecordBtn.disabled = false;
        automaticRecordBtn.disabled = false;
    }else {
        detectCamerasBtn.disabled = false;
    }
}