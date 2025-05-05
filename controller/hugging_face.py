# Importing necessary libraries and modules
import warnings  # Import the 'warnings' module for handling warnings
warnings.filterwarnings("ignore")  # Ignore warnings during execution

import gc  # Import the 'gc' module for garbage collection
import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import Pandas for data manipulation
import itertools  # Import 'itertools' for iterators and looping
from collections import Counter  # Import 'Counter' for counting elements
import matplotlib.pyplot as plt  # Import Matplotlib for data visualization
from sklearn.metrics import (  # Import various metrics from scikit-learn
    accuracy_score,  # For calculating accuracy
    roc_auc_score,  # For ROC AUC score
    confusion_matrix,  # For confusion matrix
    classification_report,  # For classification report
    f1_score  # For F1 score
)

# Import custom modules and classes
from imblearn.over_sampling import RandomOverSampler # import RandomOverSampler
import accelerate # Import the 'accelerate' module
import evaluate  # Import the 'evaluate' module
from datasets import Dataset, Image, ClassLabel  # Import custom 'Dataset', 'ClassLabel', and 'Image' classes
from transformers import (  # Import various modules from the Transformers library
    TrainingArguments,  # For training arguments
    Trainer,  # For model training
    ViTImageProcessor,  # For processing image data with ViT models
    ViTForImageClassification,  # ViT model for image classification
    DefaultDataCollator,  # For collating data in the default way
    pipeline
)
import torch  # Import PyTorch for deep learning
from torch.utils.data import DataLoader  # For creating data loaders
from torchvision.transforms import (  # Import image transformation functions
    CenterCrop,  # Center crop an image
    Compose,  # Compose multiple image transformations
    Normalize,  # Normalize image pixel values
    RandomRotation,  # Apply random rotation to images
    RandomResizedCrop,  # Crop and resize images randomly
    RandomHorizontalFlip,  # Apply random horizontal flip
    RandomAdjustSharpness,  # Adjust sharpness randomly
    Resize,  # Resize images
    ToTensor  # Convert images to PyTorch tensors
)

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

# use https://huggingface.co/docs/datasets/image_load for reference

# Import necessary libraries
image_dict = {}

# Define the list of file names
from pathlib import Path
from tqdm import tqdm
import os

from git import Repo  # Import the Repo class from gitpython

def train_model():
    # Initialize empty lists to store file names and labels
    file_names_train = []
    labels_train = []

    # Clone the dataset from the specified GitHub repository
    # Repo.clone_from("https://github.com/dilkushsingh/Facial_Emotion_Dataset.git", "facial_emotion_dataset")

    # Iterate through all image files in the specified directory
    for file in sorted((Path('facial_emotion_dataset/train_dir/').glob('*/*.*'))):
        # check number of such files in a directory
        
        label = str(file).split('\\')[-2]  # Extract the label from the file path
        labels_train.append(label)  # Add the label to the list
        file_names_train.append(str(file))  # Add the file path to the list

    # Print the total number of file names and labels
    print(len(file_names_train), len(labels_train))

    # Create a pandas dataframe from the collected file names and labels
    df_train = pd.DataFrame.from_dict({"image": file_names_train, "label": labels_train})
    print(df_train.shape)

    print(df_train.head())

    # random oversampling of minority class
    # 'y' contains the target variable (label) we want to predict
    # Extraer etiquetas
    y = df_train[['label']]

    # Eliminar columna 'label' del DataFrame de entrenamiento
    df_features = df_train.drop(['label'], axis=1)

    # Aplicar Random Oversampling
    ros = RandomOverSampler(random_state=83)
    df_features_resampled, y_resampled = ros.fit_resample(df_features, y)

    # Reconstruir el DataFrame con etiquetas balanceadas
    df_train = df_features_resampled.copy()
    df_train['label'] = y_resampled

    # Liberar memoria
    del y, df_features, y_resampled
    gc.collect()

    print("Forma después del oversampling:", df_train.shape)
    print(df_train['label'].value_counts())

    file_names_test = []
    labels_test = []

    # Clone the dataset from the specified GitHub repository
    # Repo.clone_from("https://github.com/dilkushsingh/Facial_Emotion_Dataset.git", "facial_emotion_dataset")

    # Iterate through all image files in the specified directory
    for file in sorted((Path('facial_emotion_dataset/test_dir/').glob('*/*.*'))):
        # check number of such files in a directory
        
        label = str(file).split('\\')[-2]
        labels_test.append(label)
        file_names_test.append(str(file))

    df_test = pd.DataFrame.from_dict({"image": file_names_test, "label": labels_test})
    print(df_test.shape)

    print(df_test.head())

    # # Create a dataset from a Pandas DataFrame.
    # dataset = Dataset.from_pandas(df_train).cast_column("image", Image())

    # # Display the first image in the dataset
    # dataset[0]['image'].show()

    df_train = df_train.sample(frac=0.3, random_state=42)  # Usa solo 50% del dataset de entrenamiento
    df_test = df_test.sample(frac=0.3, random_state=42)

    # Extracting the training data from the split dataset.
    train_data = df_train.copy()

    # Extracting the testing data from the split dataset.
    test_data = df_test.copy()

    # Define the pre-trained ViT model string
    model_str = 'dima806/facial_emotions_image_detection' #"google/vit-base-patch16-224-in21k"

    # Create a processor for ViT model input from the pre-trained model
    processor = ViTImageProcessor.from_pretrained(model_str)

    # Retrieve the image mean and standard deviation used for normalization
    image_mean, image_std = processor.image_mean, processor.image_std

    # Get the size (height) of the ViT model's input images
    size = processor.size["height"]
    print("Size: ", size)

    # Define a normalization transformation for the input images
    normalize = Normalize(mean=image_mean, std=image_std)

    # Define a set of transformations for training data
    _train_transforms = Compose(
        [
            Resize((size, size)),             # Resize images to the ViT model's input size
            RandomRotation(90),               # Apply random rotation
            RandomAdjustSharpness(2),         # Adjust sharpness randomly
            RandomHorizontalFlip(0.5),        # Random horizontal flip
            ToTensor(),                       # Convert images to tensors
            normalize                         # Normalize images using mean and std
        ]
    )

    # Define a set of transformations for validation data
    _val_transforms = Compose(
        [
            Resize((size, size)),             # Resize images to the ViT model's input size
            ToTensor(),                       # Convert images to tensors
            normalize                         # Normalize images using mean and std
        ]
    )

    # Define a function to apply training transformations to a batch of examples
    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(Image.open(image).convert("RGB")) for image in examples['image']]
        return examples

    # Define a function to apply validation transformations to a batch of examples
    def val_transforms(examples):
        
        examples['pixel_values'] = [_val_transforms(Image.open(image).convert("RGB")) for image in examples['image']]
        return examples

    # Convert train_data and test_data DataFrames to Hugging Face Dataset objects
    train_data = Dataset.from_pandas(train_data)
    test_data = Dataset.from_pandas(test_data)

    # Set the transforms for the training data
    train_data = train_data.with_transform(train_transforms)

    # Set the transforms for the test/validation data
    test_data = test_data.with_transform(val_transforms)

    # Define a collate function that prepares batched data for model training.
    def collate_fn(examples):
        # Stack the pixel values from individual examples into a single tensor.
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        
        # Convert the label strings in examples to corresponding numeric IDs using label2id dictionary.
        labels = torch.tensor([label2id[example['label']] for example in examples])
        
        # Return a dictionary containing the batched pixel values and labels.
        return {"pixel_values": pixel_values, "labels": labels}


    # Create a list of unique labels by converting 'labels' to a set and then back to a list
    labels_list = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy'] # list(set(labels))

    # Initialize empty dictionaries to map labels to IDs and vice versa
    label2id, id2label = dict(), dict()

    # Iterate over the unique labels and assign each label an ID, and vice versa
    for i, label in enumerate(labels_list):
        label2id[label] = i  # Map the label to its corresponding ID
        id2label[i] = label  # Map the ID to its corresponding label

    # Print the resulting dictionaries for reference
    print("Mapping of IDs to Labels:", id2label, '\n')
    print("Mapping of Labels to IDs:", label2id)

    # Create a ViTForImageClassification model from a pretrained checkpoint with a specified number of output labels.
    model = ViTForImageClassification.from_pretrained(model_str, num_labels=len(labels_list))

    # Configure the mapping of class labels to their corresponding indices for later reference.
    model.config.id2label = id2label
    model.config.label2id = label2id

    # Calculate and print the number of trainable parameters in millions for the model.
    print(model.num_parameters(only_trainable=True) / 1e6)

    #  Load the accuracy metric from a module named 'evaluate'
    accuracy = evaluate.load("accuracy")

    # Define a function 'compute_metrics' to calculate evaluation metrics
    def compute_metrics(eval_pred):
        # Extract model predictions from the evaluation prediction object
        predictions = eval_pred.predictions
        
        # Extract true labels from the evaluation prediction object
        label_ids = eval_pred.label_ids
        
        # Calculate accuracy using the loaded accuracy metric
        # Convert model predictions to class labels by selecting the class with the highest probability (argmax)
        predicted_labels = predictions.argmax(axis=1)
        
        # Calculate accuracy score by comparing predicted labels to true labels
        acc_score = accuracy.compute(predictions=predicted_labels, references=label_ids)['accuracy']
        
        # Return the computed accuracy as a dictionary with the key "accuracy"
        return {
            "accuracy": acc_score
        }
        
    # Define the name of the evaluation metric to be used during training and evaluation.
    metric_name = "accuracy"

    # Define the name of the model, which will be used to create a directory for saving model checkpoints and outputs.
    model_name = "facial_emotions_image_detection"

    # Define the number of training epochs for the model.
    num_train_epochs = 3

    # Create an instance of TrainingArguments to configure training settings.
    args = TrainingArguments(
        # Specify the directory where model checkpoints and outputs will be saved.
        output_dir=model_name,
        
        # Specify the directory where training logs will be stored.
        logging_dir='./logs',
        
        fp16=True, # Enable mixed precision training (if supported by the hardware).
        
        # Define the evaluation strategy, which is performed at the end of each epoch.
        eval_strategy="no",
        
        # Set the learning rate for the optimizer.
        learning_rate=5e-5,
        
        # Define the batch size for training on each device.
        per_device_train_batch_size=64,
        
        # Define the batch size for evaluation on each device.
        per_device_eval_batch_size=8,
        
        # Specify the total number of training epochs.
        num_train_epochs=num_train_epochs,
        
        # Apply weight decay to prevent overfitting.
        weight_decay=0.02,
        
        # Set the number of warm-up steps for the learning rate scheduler.
        warmup_steps=50,
        
        # Disable the removal of unused columns from the dataset.
        remove_unused_columns=False,
        
        # Define the strategy for saving model checkpoints (per epoch in this case).
        save_strategy='no',
        
        # Load the best model at the end of training.
        load_best_model_at_end=False,
        
        # Limit the total number of saved checkpoints to save space.
        save_total_limit=1,
        
        # Specify that training progress should not be reported.
        report_to="none"
    )

    # Create a Trainer instance for fine-tuning a language model.

    # - `model`: The pre-trained language model to be fine-tuned.
    # - `args`: Configuration settings and hyperparameters for training.
    # - `train_dataset`: The dataset used for training the model.
    # - `eval_dataset`: The dataset used for evaluating the model during training.
    # - `data_collator`: A function that defines how data batches are collated and processed.
    # - `compute_metrics`: A function for computing custom evaluation metrics.
    # - `tokenizer`: The tokenizer used for processing text data.

    trainer = Trainer(
        model,
        args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    # Evaluate the pre-training model's performance on a test dataset.
    # This function calculates various metrics such as accuracy, loss, etc.,
    # to assess how well the model is performing on unseen data.

    trainer.evaluate()

    # Start training the model using the trainer object.
    trainer.train()

    # Evaluate the post-training model's performance on the validation or test dataset.
    # This function computes various evaluation metrics like accuracy, loss, etc.
    # and provides insights into how well the model is performing.

    trainer.evaluate()

    # Use the trained 'trainer' to make predictions on the 'test_data'.
    outputs = trainer.predict(test_data)

    # Print the metrics obtained from the prediction outputs.
    print(outputs.metrics)

    # Extract the true labels from the model outputs
    y_true = outputs.label_ids

    # Predict the labels by selecting the class with the highest probability
    y_pred = outputs.predictions.argmax(1)

    # Define a function to plot a confusion matrix
    def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(10, 8)):
        """
        This function plots a confusion matrix.

        Parameters:
            cm (array-like): Confusion matrix as returned by sklearn.metrics.confusion_matrix.
            classes (list): List of class names, e.g., ['Class 0', 'Class 1'].
            title (str): Title for the plot.
            cmap (matplotlib colormap): Colormap for the plot.
        """
        # Create a figure with a specified size
        plt.figure(figsize=figsize)
        
        # Display the confusion matrix as an image with a colormap
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        # Define tick marks and labels for the classes on the axes
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        fmt = '.0f'
        # Add text annotations to the plot indicating the values in the cells
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        # Label the axes
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # Ensure the plot layout is tight
        plt.tight_layout()
        # Display the plot
        plt.show()

    # Calculate accuracy and F1 score
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    # Display accuracy and F1 score
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Get the confusion matrix if there are a small number of labels
    if len(labels_list) <= 150:
        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot the confusion matrix using the defined function
        plot_confusion_matrix(cm, labels_list, figsize=(8, 6))
        
    # Finally, display classification report
    print()
    print("Classification report:")
    print()
    print(classification_report(y_true, y_pred, target_names=labels_list, digits=4))

    # Save the trained model: This line of code is responsible for saving the model
    # that has been trained using the trainer object. It will serialize the model
    # and its associated weights, making it possible to reload and use the model
    # in the future without the need to retrain it.
    trainer.save_model("facial_emotions_image_detection")
    processor.save_pretrained("facial_emotions_image_detection")

def classify_image(image_path):
    array_emotions = []
    
    # Cargar modelo y preprocesador desde carpeta local
    model = ViTForImageClassification.from_pretrained("facial_emotions_image_detection", local_files_only=True)
    processor = ViTImageProcessor.from_pretrained("facial_emotions_image_detection", local_files_only=True)

    # Crear el pipeline
    pipe = pipeline("image-classification", model=model, image_processor=processor, device=0)  # device=0 para GPU, usa -1 para CPU
    
    # Clasificar imagen
    for filename in os.listdir(image_path):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            image = Image.open(os.path.join(image_path, filename)).convert("RGB")
            results = pipe(image)
            filename_splited = filename.split('_')[0]

            # Mostrar resultados
            print(f"Results for {filename_splited}:")
            for result in results:
                print(f"Emotion: {result['label']}, %: {result['score']:.4f}")
                
                # Find the emotion with the highest score
                highest_emotion = max(results, key=lambda x: x['score'])
            img_url = f"/face_images/{filename}"
            array_emotions.append(f"<img src='{img_url}' width='80'> {filename_splited}: {highest_emotion['label']}<br>")
            print(f"Highest Emotion: {highest_emotion['label']}, Score: {highest_emotion['score']:.4f}")
            print()
    return array_emotions