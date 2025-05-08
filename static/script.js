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
    fileInput.value = '';
    responseMessage.textContent = '';
    output.textContent = '';
    jokesMessage.textContent = '';
    preview.innerHTML = '<p>No image loaded</p>';
    const btn = document.getElementById('submit-changes');
    if (btn) btn.remove();
    positive_percentage.textContent = '';
    negative_percentage.textContent = '';
});

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
      submitButton.disabled = true;
      resetButton.disabled = true;
      const res = await fetch('/process', {
          method: 'POST',
          body: formData,
      });

      const data = await res.json();

      if (res.ok) {
          console.log(data);
          output.innerHTML = data.join("");
          jokesMessage.textContent = '';
          responseMessage.textContent = 'Image processed successfully!';
          submitButton.disabled = false;
          resetButton.disabled = false;

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

      } else {
          responseMessage.textContent = data.error || 'Unknown error';
          output.textContent = '';
          submitButton.disabled = false;
          resetButton.disabled = false;
      }

  } catch (err) {
      responseMessage.textContent = 'Error sending image';
      console.error(err);
      submitButton.disabled = false;
      resetButton.disabled = false;
  }
});

show_percentages = (emotions, puntuation_emotions) => {
    // show the percentatge of all the emotions with the array of puntuation_emotions
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