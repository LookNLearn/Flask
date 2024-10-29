import io
import base64
from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from flask_cors import CORS
from facenet_pytorch import MTCNN
from torchvision import models
import torch.nn as nn
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(keep_all=True, device=device)

model_ft = models.efficientnet_b0(pretrained=True)
num_ftrs = model_ft.classifier[1].in_features
model_ft.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, 7)
)
checkpoint_path = './checkpoints/best_checkpoint_mtcnn.pth'

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_ft.load_state_dict(checkpoint['model_state_dict'], strict=False)
model_ft = model_ft.to(device)
model_ft.eval()

CATEGORIES = ['angry', 'anxiety', 'embarrass', 'hurt', 'joy', 'neutral', 'sad']

def transform_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(device)

def predict_emotion(face_img):
    face_tensor = transform_image(face_img)
    with torch.no_grad():
        outputs = model_ft(face_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
    predictions = sorted(zip(probabilities, CATEGORIES), reverse=True, key=lambda x: x[0])
    return predictions

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
    except UnidentifiedImageError:
        return jsonify({'error': 'Cannot identify image file'}), 400

    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        return jsonify({'error': 'No face detected'}), 400

    results = []
    for box in boxes:
        face = img.crop((box[0], box[1], box[2], box[3]))
        predictions = predict_emotion(face)
        results.append({
            'box': box.tolist(),
            'predictions': [{'label': label, 'probability': float(prob)} for prob, label in predictions]
        })

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({'results': results, 'image': img_str})

if __name__ == '__main__':
    app.run(debug=True)
