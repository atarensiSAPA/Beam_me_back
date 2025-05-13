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
submitRecordBtn.disabled = true;

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
        } else {
            submitRecordBtn.disabled = true;
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
    positive_percentage.textContent = '';
    negative_percentage.textContent = '';
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        video.srcObject = null;
        videoDevices = [];
        cameraSelect.innerHTML = '';
        submitRecordBtn.disabled = true;
    }
}

// Interceptar el envío del formulario
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
    // if want to reset the output when the user submit a new image
    // jokesMessage.textContent = '';
    // output.textContent = '';
    // positive_percentage.textContent = '';
    // negative_percentage.textContent = '';
    disableButtons();
    const res = await fetch('/process', {
    method: 'POST',
    body: formData,
    });

    const data = await res.json();

    if (res.ok) {
        console.log(data);
        output.innerHTML = data.join("");
        response_function(data);

    } else {
        responseMessage.textContent = data.error || 'Unknown error';
        output.textContent = '';
        jokesMessage.textContent = '';
        positive_percentage.textContent = '';
        negative_percentage.textContent = '';
        enableButtons();
    }

  } catch (err) {
      responseMessage.textContent = 'Error sending image';
      console.error(err);
      enableButtons();
  }
});

show_percentages = (emotions, puntuation_emotions) => {
    // show the percentage of all the emotions with the array of puntuation_emotions
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
    
        document.getElementById('output-wrapper').appendChild(submitChangesBtn);
    
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

detectCamerasBtn.addEventListener('click', () => {
    getCameras();
});

submitRecordBtn.addEventListener('click', async () => {
    if (stream) {
        const track = stream.getVideoTracks()[0];
        const imageCapture = new ImageCapture(track);
        try {
            responseMessage.textContent = 'Processing image from camera...';
            const blob = await imageCapture.takePhoto();
            const file = new File([blob], 'captured_image.jpg', { type: 'image/jpeg' });
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            fileInput.files = dataTransfer.files;
            const formData = new FormData();
            formData.append('image', file);

            const record_res = await fetch('/process', {
            method: 'POST',
            body: formData,
            });
            const record_data = await record_res.json();
            if (record_data.ok) {
                console.log(record_data);
                output.innerHTML = data.join("");
                response_function(record_data);

            } else {
                responseMessage.textContent = record_data.error || 'Unknown error';
                output.textContent = '';
                jokesMessage.textContent = '';
                positive_percentage.textContent = '';
                negative_percentage.textContent = '';
                enableButtons();
            }
        } catch (err) {
            console.error("Error capturing photo:", err);
            console.error(err);
            enableButtons();
        }
    } else {
        alert("No camera stream available.");
    }
});

const response_function = async (data) => {
    jokesMessage.textContent = '';
    responseMessage.textContent = 'Image processed successfully!';
    enableButtons();

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
    resetButton.disabled = true;
    submitRecordBtn.disabled = true;
}

function enableButtons() {
    submitButton.disabled = false;
    resetButton.disabled = false;
    if (videoDevices.length > 0) {
        submitRecordBtn.disabled = false;
    }
}