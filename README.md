## Emotion Classification Flask Application
This repository contains a Flask-based REST API that performs emotion classification from face images using an EfficientNet-B0 model. 
The application detects faces in an uploaded image, processes each detected face, and predicts the emotion for each face.

## Features
*Face Detection*: Uses MTCNN for detecting faces in the uploaded image.
*Emotion Classification*: Classifies emotions into the following categories:
angry
disgust
fear
happy
neutral
sad
surprise
*Pre-trained Model*: Loads a pre-trained EfficientNet-B0 model from an *S3 bucket*.
*API Endpoints*: Provides an endpoint for uploading images and receiving predictions.
*CORS Support*: Enabled CORS to allow requests from different origins.

## Requirements
### Prerequisites
Python 3.8+
Dependencies listed in requirements.txt (to be created)
### Python Libraries Used
Flask: Web framework for serving the API.
facenet-pytorch: For face detection using MTCNN.
torch: PyTorch for model inference.
torchvision: For pre-defined models and transformations.
Pillow: For image processing.
requests: For downloading the model from S3.
flask-cors: For enabling CORS.

## Setup Instructions
### 1. Clone the Repository
bash
>> git clone https://github.com/your-repo/emotion-classification-flask.git
>> cd emotion-classification-flask

### 2. Install Dependencies
Create a virtual environment and install the required dependencies.
>> python -m venv venv
>> source venv/bin/activate  # For Windows: venv\Scripts\activate
>> pip install -r requirements.txt

### 3. Set Up the Model
The application downloads the model from an S3 bucket upon the first run. Ensure the S3_URL variable in the code points to the correct model URL.

### 4. Run the Application
Run the Flask application on 0.0.0.0:8080.
>> python app.py

## Internal Workflow
### 1. Image Upload:
- Validates the image file from the request.
- Loads the image using PIL.

### 2. Face Detection:
- Uses MTCNN to detect faces in the image.
- Extracts each face region for analysis.

### 3. Emotion Prediction:
- Applies preprocessing transformations.
- Predicts emotion probabilities using the EfficientNet-B0 model.

### 4. Response:
- Returns predictions for each detected face.

## Model Details
*Architecture*: EfficientNet-B0
*Fine-tuned for*: 7-class emotion classification
*Face Detection*: MTCNN
*Input Size*: 224x224
*Normalization*: Mean=[0.5, 0.5, 0.5], Std=[0.5, 0.5, 0.5]
