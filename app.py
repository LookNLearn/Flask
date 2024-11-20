# # # # # # import io
# # # # # # from flask import Flask, request, jsonify
# # # # # # import torch
# # # # # # from torchvision import transforms
# # # # # # from PIL import Image, UnidentifiedImageError
# # # # # # from flask_cors import CORS
# # # # # # from facenet_pytorch import MTCNN
# # # # # # from torchvision import models
# # # # # # import torch.nn as nn
# # # # # # import os

# # # # # # app = Flask(__name__)
# # # # # # # app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB 제한 설정
# # # # # # CORS(app, resources={r"/*": {"origins": "*"}})

# # # # # # device = torch.device("cpu")

# # # # # # mtcnn = MTCNN(keep_all=True, device=device)

# # # # # # model_ft = models.efficientnet_b0()
# # # # # # num_ftrs = model_ft.classifier[1].in_features
# # # # # # model_ft.classifier = nn.Sequential(
# # # # # #     nn.Dropout(p=0.5),
# # # # # #     nn.Linear(num_ftrs, 7)
# # # # # # )
# # # # # # checkpoint_path = './checkpoints/best_checkpoint_mtcnn.pth'

# # # # # # if os.path.exists(checkpoint_path):
# # # # # #     checkpoint = torch.load(checkpoint_path, map_location=device)
# # # # # #     model_ft.load_state_dict(checkpoint['model_state_dict'], strict=False)
# # # # # # model_ft = model_ft.to(device)
# # # # # # model_ft.eval()

# # # # # # CATEGORIES = ['angry', 'anxiety', 'embarrass', 'hurt', 'joy', 'neutral', 'sad']

# # # # # # def transform_image(img):
# # # # # #     transform = transforms.Compose([
# # # # # #         transforms.Resize((224, 224)),
# # # # # #         transforms.ToTensor(),
# # # # # #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# # # # # #     ])
# # # # # #     return transform(img).unsqueeze(0).to(device)

# # # # # # def predict_emotion(face_img):
# # # # # #     face_tensor = transform_image(face_img)
# # # # # #     with torch.no_grad():
# # # # # #         outputs = model_ft(face_tensor)
# # # # # #         probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
# # # # # #     predictions = sorted(zip(probabilities, CATEGORIES), reverse=True, key=lambda x: x[0])
# # # # # #     return predictions

# # # # # # @app.route('/upload', methods=['POST'])
# # # # # # def upload_file():
# # # # # #     if 'image' not in request.files:
# # # # # #         return jsonify({'error': 'No image part in the request'}), 400

# # # # # #     file = request.files['image']
# # # # # #     if file.filename == '':
# # # # # #         return jsonify({'error': 'No selected file'}), 400

# # # # # #     try:
# # # # # #         img = Image.open(file.stream).convert('RGB')
# # # # # #     except UnidentifiedImageError:
# # # # # #         return jsonify({'error': 'Cannot identify image file'}), 400

# # # # # #     # 얼굴 탐지
# # # # # #     boxes, _ = mtcnn.detect(img)
# # # # # #     if boxes is None:
# # # # # #         return jsonify({'error': 'No face detected'}), 400

# # # # # #     # 예측 정보만 포함한 결과 반환
# # # # # #     results = []
# # # # # #     for box in boxes:
# # # # # #         face = img.crop((box[0], box[1], box[2], box[3]))
# # # # # #         predictions = predict_emotion(face)
# # # # # #         results.append({
# # # # # #             'box': box.tolist(),
# # # # # #             'predictions': [{'label': label, 'probability': float(prob)} for prob, label in predictions]
# # # # # #         })

# # # # # #     return jsonify({
# # # # # #         'results': results
# # # # # #     })

# # # # # # if __name__ == '__main__':
# # # # # #     app.run(debug=True)

# # # # # import io
# # # # # from flask import Flask, request, jsonify
# # # # # import torch
# # # # # from torchvision import transforms
# # # # # from PIL import Image, UnidentifiedImageError
# # # # # from flask_cors import CORS
# # # # # from facenet_pytorch import MTCNN
# # # # # from torchvision import models
# # # # # import torch.nn as nn
# # # # # import os

# # # # # app = Flask(__name__)
# # # # # CORS(app, resources={r"/*": {"origins": "*"}})

# # # # # device = torch.device("cpu")

# # # # # mtcnn = MTCNN(keep_all=True, device=device)

# # # # # model_ft = models.efficientnet_b0()
# # # # # num_ftrs = model_ft.classifier[1].in_features
# # # # # model_ft.classifier = nn.Sequential(
# # # # #     nn.Dropout(p=0.5),
# # # # #     nn.Linear(num_ftrs, 7)
# # # # # )
# # # # # checkpoint_path = './checkpoints/best_checkpoint_mtcnn.pth'

# # # # # if os.path.exists(checkpoint_path):
# # # # #     checkpoint = torch.load(checkpoint_path, map_location=device)
# # # # #     model_ft.load_state_dict(checkpoint['model_state_dict'], strict=False)
# # # # # model_ft = model_ft.to(device)
# # # # # model_ft.eval()

# # # # # CATEGORIES = ['angry', 'anxiety', 'embarrass', 'hurt', 'joy', 'neutral', 'sad']

# # # # # def resize_image(img, max_size=800):
# # # # #     """
# # # # #     이미지를 최대 max_size로 리사이즈합니다.
# # # # #     img: PIL 이미지 객체
# # # # #     max_size: 이미지의 긴 쪽의 최대 픽셀 크기
# # # # #     """
# # # # #     width, height = img.size
# # # # #     if max(width, height) > max_size:
# # # # #         scaling_factor = max_size / float(max(width, height))
# # # # #         new_size = (int(width * scaling_factor), int(height * scaling_factor))
# # # # #         img = img.resize(new_size, Image.ANTIALIAS)
# # # # #     return img

# # # # # def transform_image(img):
# # # # #     transform = transforms.Compose([
# # # # #         transforms.Resize((224, 224)),
# # # # #         transforms.ToTensor(),
# # # # #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# # # # #     ])
# # # # #     return transform(img).unsqueeze(0).to(device)

# # # # # def predict_emotion(face_img):
# # # # #     face_tensor = transform_image(face_img)
# # # # #     with torch.no_grad():
# # # # #         outputs = model_ft(face_tensor)
# # # # #         probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
# # # # #     predictions = sorted(zip(probabilities, CATEGORIES), reverse=True, key=lambda x: x[0])
# # # # #     return predictions

# # # # # @app.route('/upload', methods=['POST'])
# # # # # def upload_file():
# # # # #     if 'image' not in request.files:
# # # # #         return jsonify({'error': 'No image part in the request'}), 400

# # # # #     file = request.files['image']
# # # # #     if file.filename == '':
# # # # #         return jsonify({'error': 'No selected file'}), 400

# # # # #     try:
# # # # #         img = Image.open(file.stream).convert('RGB')
# # # # #     except UnidentifiedImageError:
# # # # #         return jsonify({'error': 'Cannot identify image file'}), 400

# # # # #     # 이미지 크기 줄이기
# # # # #     img = resize_image(img, max_size=800)

# # # # #     # 얼굴 탐지
# # # # #     boxes, _ = mtcnn.detect(img)
# # # # #     if boxes is None:
# # # # #         return jsonify({'error': 'No face detected'}), 400

# # # # #     # 예측 정보만 포함한 결과 반환
# # # # #     results = []
# # # # #     for box in boxes:
# # # # #         face = img.crop((box[0], box[1], box[2], box[3]))
# # # # #         predictions = predict_emotion(face)
# # # # #         results.append({
# # # # #             'box': box.tolist(),
# # # # #             'predictions': [{'label': label, 'probability': float(prob)} for prob, label in predictions]
# # # # #         })

# # # # #     return jsonify({
# # # # #         'results': results
# # # # #     })

# # # # # # if __name__ == '__main__':
# # # # # #     app.run(debug=True)

# # # # import io
# # # # from flask import Flask, request, jsonify
# # # # import torch
# # # # from torchvision import transforms
# # # # from PIL import Image, UnidentifiedImageError
# # # # from flask_cors import CORS
# # # # from facenet_pytorch import MTCNN
# # # # from torchvision import models
# # # # import torch.nn as nn
# # # # import os

# # # # app = Flask(__name__)
# # # # CORS(app, resources={r"/*": {"origins": "*"}})

# # # # device = torch.device("cpu")

# # # # mtcnn = MTCNN(keep_all=True, device=device)

# # # # model_ft = models.efficientnet_b0()
# # # # num_ftrs = model_ft.classifier[1].in_features
# # # # model_ft.classifier = nn.Sequential(
# # # #     nn.Dropout(p=0.5),
# # # #     nn.Linear(num_ftrs, 7)
# # # # )
# # # # checkpoint_path = './checkpoints/best_checkpoint_mtcnn.pth'

# # # # if os.path.exists(checkpoint_path):
# # # #     checkpoint = torch.load(checkpoint_path, map_location=device)
# # # #     model_ft.load_state_dict(checkpoint['model_state_dict'], strict=False)
# # # # model_ft = model_ft.to(device)
# # # # model_ft.eval()

# # # # CATEGORIES = ['angry', 'anxiety', 'embarrass', 'hurt', 'joy', 'neutral', 'sad']

# # # # def resize_image(img, max_size=800, quality=70):
# # # #     """
# # # #     이미지를 최대 max_size로 리사이즈하고 JPEG 압축 품질을 조정합니다.
# # # #     img: PIL 이미지 객체
# # # #     max_size: 이미지의 긴 쪽의 최대 픽셀 크기
# # # #     quality: JPEG 압축 품질 (1-100, 낮을수록 압축률이 높음)
# # # #     """
# # # #     width, height = img.size
# # # #     if max(width, height) > max_size:
# # # #         scaling_factor = max_size / float(max(width, height))
# # # #         new_size = (int(width * scaling_factor), int(height * scaling_factor))
# # # #         img = img.resize(new_size, Image.LANCZOS)

# # # #     # 압축 품질을 조정하여 메모리 내에 JPEG 형식으로 저장
# # # #     buffer = io.BytesIO()
# # # #     img.save(buffer, format="JPEG", quality=quality)
# # # #     buffer.seek(0)
# # # #     return Image.open(buffer)

# # # # def transform_image(img):
# # # #     transform = transforms.Compose([
# # # #         transforms.Resize((224, 224)),
# # # #         transforms.ToTensor(),
# # # #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# # # #     ])
# # # #     return transform(img).unsqueeze(0).to(device)

# # # # def predict_emotion(face_img):
# # # #     face_tensor = transform_image(face_img)
# # # #     with torch.no_grad():
# # # #         outputs = model_ft(face_tensor)
# # # #         probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
# # # #     predictions = sorted(zip(probabilities, CATEGORIES), reverse=True, key=lambda x: x[0])
# # # #     return predictions

# # # # @app.route('/upload', methods=['POST'])
# # # # def upload_file():
# # # #     if 'image' not in request.files:
# # # #         return jsonify({'error': 'No image part in the request'}), 400

# # # #     file = request.files['image']
# # # #     if file.filename == '':
# # # #         return jsonify({'error': 'No selected file'}), 400

# # # #     try:
# # # #         img = Image.open(file.stream).convert('RGB')
# # # #     except UnidentifiedImageError:
# # # #         return jsonify({'error': 'Cannot identify image file'}), 400

# # # #     # 이미지 크기 및 압축 조정
# # # #     img = resize_image(img, max_size=800, quality=70)

# # # #     # 얼굴 탐지
# # # #     boxes, _ = mtcnn.detect(img)
# # # #     if boxes is None:
# # # #         return jsonify({'error': 'No face detected'}), 400

# # # #     # 예측 정보만 포함한 결과 반환
# # # #     results = []
# # # #     for box in boxes:
# # # #         face = img.crop((box[0], box[1], box[2], box[3]))
# # # #         predictions = predict_emotion(face)
# # # #         results.append({
# # # #             'box': box.tolist(),
# # # #             'predictions': [{'label': label, 'probability': float(prob)} for prob, label in predictions]
# # # #         })

# # # #     return jsonify({
# # # #         'results': results
# # # #     })

# # # # if __name__ == '__main__':
# # # #     app.run(debug=True)

# # # import io
# # # from flask import Flask, request, jsonify
# # # import torch
# # # from torchvision import transforms
# # # from PIL import Image, UnidentifiedImageError
# # # from flask_cors import CORS
# # # from facenet_pytorch import MTCNN
# # # from torchvision import models
# # # import torch.nn as nn
# # # import os

# # # app = Flask(__name__)
# # # CORS(app, resources={r"/*": {"origins": "*"}})

# # # device = torch.device("cpu")

# # # # MTCNN 얼굴 탐지 모델 초기화
# # # mtcnn = MTCNN(keep_all=True, device=device)

# # # # EfficientNet-B0 모델 불러오기 및 수정
# # # model_ft = models.efficientnet_b0()
# # # num_ftrs = model_ft.classifier[1].in_features
# # # model_ft.classifier = nn.Sequential(
# # #     nn.Dropout(p=0.5),
# # #     nn.Linear(num_ftrs, 7)
# # # )
# # # checkpoint_path = './checkpoints/best_checkpoint_mtcnn.pth'

# # # # 학습된 가중치 로드
# # # if os.path.exists(checkpoint_path):
# # #     checkpoint = torch.load(checkpoint_path, map_location=device)
# # #     model_ft.load_state_dict(checkpoint['model_state_dict'], strict=False)
# # #     print("Model weights loaded successfully")
# # # else:
# # #     print("Checkpoint path does not exist")

# # # model_ft = model_ft.to(device)
# # # model_ft.eval()

# # # # 감정 분류 카테고리
# # # CATEGORIES = ['angry', 'anxiety', 'embarrass', 'hurt', 'joy', 'neutral', 'sad']

# # # # 이미지를 112x112로 리사이즈하는 함수
# # # def resize_image(img, target_size=(112, 112)):
# # #     """
# # #     이미지를 target_size로 리사이즈합니다.
# # #     img: PIL 이미지 객체
# # #     target_size: (너비, 높이) 튜플
# # #     """
# # #     return img.resize(target_size, Image.LANCZOS)

# # # # 이미지 전처리 함수
# # # def transform_image(img):
# # #     transform = transforms.Compose([
# # #         transforms.Resize((224, 224)),  # 224x224로 리사이즈 (EfficientNet의 기본 입력 크기)
# # #         transforms.ToTensor(),
# # #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# # #     ])
# # #     return transform(img).unsqueeze(0).to(device)

# # # # 감정 예측 함수
# # # def predict_emotion(face_img):
# # #     face_tensor = transform_image(face_img)
# # #     with torch.no_grad():
# # #         outputs = model_ft(face_tensor)
# # #         probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
# # #     predictions = sorted(zip(probabilities, CATEGORIES), reverse=True, key=lambda x: x[0])
# # #     return predictions

# # # # 파일 업로드 엔드포인트
# # # @app.route('/upload', methods=['POST'])
# # # def upload_file():
# # #     if 'image' not in request.files:
# # #         return jsonify({'error': 'No image part in the request'}), 400

# # #     file = request.files['image']
# # #     if file.filename == '':
# # #         return jsonify({'error': 'No selected file'}), 400

# # #     try:
# # #         img = Image.open(file.stream).convert('RGB')
# # #     except UnidentifiedImageError:
# # #         return jsonify({'error': 'Cannot identify image file'}), 400

# # #     # 이미지 크기를 112x112로 리사이즈
# # #     img = resize_image(img, target_size=(112, 112))

# # #     # 얼굴 탐지
# # #     boxes, _ = mtcnn.detect(img)
# # #     if boxes is None:
# # #         return jsonify({'error': 'No face detected'}), 400

# # #     # 얼굴 영역만을 예측하여 결과 반환
# # #     results = []
# # #     for box in boxes:
# # #         face = img.crop((box[0], box[1], box[2], box[3]))
# # #         predictions = predict_emotion(face)
# # #         results.append({
# # #             'box': box.tolist(),
# # #             'predictions': [{'label': label, 'probability': float(prob)} for prob, label in predictions]
# # #         })

# # #     return jsonify({
# # #         'results': results
# # #     })

# # # if __name__ == '__main__':
# # #     app.run(debug=True)

# # #=============================================================================================================#
# # # import io
# # # from flask import Flask, request, jsonify
# # # import torch
# # # from torchvision import transforms
# # # from PIL import Image, UnidentifiedImageError
# # # from flask_cors import CORS
# # # from facenet_pytorch import MTCNN
# # # from torchvision import models
# # # import torch.nn as nn
# # # import os

# # # app = Flask(__name__)
# # # CORS(app, resources={r"/*": {"origins": "*"}})

# # # device = torch.device("cpu")

# # # # MTCNN 얼굴 탐지 모델 초기화
# # # mtcnn = MTCNN(keep_all=True, device=device)

# # # # EfficientNet-B0 모델 불러오기 및 수정
# # # model_ft = models.efficientnet_b0()
# # # num_ftrs = model_ft.classifier[1].in_features
# # # model_ft.classifier = nn.Sequential(
# # #     nn.Dropout(p=0.5),
# # #     nn.Linear(num_ftrs, 7)
# # # )

# # # # .pth 파일에서 가중치 로드
# # # checkpoint_path = './checkpoints/best_checkpoint_mtcnn.pth'
# # # if os.path.exists(checkpoint_path):
# # #     checkpoint = torch.load(checkpoint_path, map_location=device)
# # #     model_ft.load_state_dict(checkpoint['model_state_dict'], strict=False)
# # #     print("Model weights loaded successfully")
# # # else:
# # #     print("Checkpoint path does not exist")

# # # CATEGORIES = ['angry', 'anxiety', 'embarrass', 'hurt', 'joy', 'neutral', 'sad']

# # # def transform_image(img):
# # #     transform = transforms.Compose([
# # #         transforms.Resize((224, 224)),
# # #         transforms.ToTensor(),
# # #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# # #     ])
# # #     return transform(img).unsqueeze(0).to(device)

# # # def predict_emotion(face_img):
# # #     face_tensor = transform_image(face_img)
# # #     with torch.no_grad():
# # #         outputs = model_ft(face_tensor)
# # #         probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
# # #     predictions = sorted(zip(probabilities, CATEGORIES), reverse=True, key=lambda x: x[0])
# # #     return predictions

# # # @app.route('/upload', methods=['POST'])
# # # def upload_file():
# # #     if 'image' not in request.files:
# # #         return jsonify({'error': 'No image part in the request'}), 400

# # #     file = request.files['image']
# # #     if file.filename == '':
# # #         return jsonify({'error': 'No selected file'}), 400

# # #     try:
# # #         img = Image.open(file.stream).convert('RGB')
# # #     except UnidentifiedImageError:
# # #         return jsonify({'error': 'Cannot identify image file'}), 400

# # #     # 원본 이미지에서 얼굴 탐지
# # #     boxes, _ = mtcnn.detect(img)
# # #     if boxes is None:
# # #         return jsonify({'error': 'No face detected'}), 400

# # #     # 얼굴 영역만을 예측하여 결과 반환
# # #     results = []
# # #     for box in boxes:
# # #         face = img.crop((box[0], box[1], box[2], box[3]))  # 얼굴 영역 추출
# # #         predictions = predict_emotion(face)  # 감정 예측
# # #         results.append({
# # #             'predictions': [{'label': label, 'probability': float(prob)} for prob, label in predictions]
# # #         })

# # #     return jsonify({'results': results})

# # # # if __name__ == '__main__':
# # # #     app.run(debug=True)



# # #===========================================================================================================#
# # import io
# # from flask import Flask, request, jsonify
# # import torch
# # from torchvision import transforms
# # from PIL import Image, UnidentifiedImageError
# # from flask_cors import CORS
# # from facenet_pytorch import MTCNN
# # from torchvision import models
# # import torch.nn as nn
# # import os

# # app = Flask(__name__)
# # CORS(app, resources={r"/*": {"origins": "*"}})

# # device = torch.device("cpu")

# # # MTCNN 얼굴 탐지 모델 초기화
# # mtcnn = MTCNN(keep_all=True, device=device)

# # # EfficientNet-B0 모델 정의 및 수정
# # model_ft = models.efficientnet_b0()
# # num_ftrs = model_ft.classifier[1].in_features
# # model_ft.classifier = nn.Sequential(
# #     nn.Dropout(p=0.5),
# #     nn.Linear(num_ftrs, 7)
# # )

# # # 양자화된 모델 로드
# # quantized_checkpoint_path = './checkpoints/quantized_model.pth'
# # if os.path.exists(quantized_checkpoint_path):
# #     print("Loading quantized model...")
# #     # 양자화된 모델을 동적 양자화로 변환 후 가중치 로드
# #     model_ft = torch.quantization.quantize_dynamic(
# #         model_ft, {torch.nn.Linear}, dtype=torch.qint8
# #     )
# #     model_ft.load_state_dict(torch.load(quantized_checkpoint_path, map_location=device))
# #     print("Quantized model loaded successfully.")
# # else:
# #     # 양자화된 모델이 없을 경우 기존 모델 로드
# #     print("Quantized model not found. Loading regular model.")
# #     checkpoint_path = './checkpoints/best_checkpoint_mtcnn.pth'
# #     if os.path.exists(checkpoint_path):
# #         checkpoint = torch.load(checkpoint_path, map_location=device)
# #         model_ft.load_state_dict(checkpoint['model_state_dict'], strict=False)
# #         print("Regular model weights loaded successfully.")
# #     else:
# #         print("Checkpoint path does not exist.")

# # model_ft = model_ft.to(device)
# # model_ft.eval()

# # CATEGORIES = ['angry', 'anxiety', 'embarrass', 'hurt', 'joy', 'neutral', 'sad']

# # def transform_image(img):
# #     transform = transforms.Compose([
# #         transforms.Resize((224, 224)),
# #         transforms.ToTensor(),
# #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# #     ])
# #     return transform(img).unsqueeze(0).to(device)

# # def predict_emotion(face_img):
# #     face_tensor = transform_image(face_img)
# #     with torch.no_grad():
# #         outputs = model_ft(face_tensor)
# #         probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
# #     predictions = sorted(zip(probabilities, CATEGORIES), reverse=True, key=lambda x: x[0])
# #     return predictions

# # @app.route('/upload', methods=['POST'])
# # def upload_file():
# #     if 'image' not in request.files:
# #         return jsonify({'error': 'No image part in the request'}), 400

# #     file = request.files['image']
# #     if file.filename == '':
# #         return jsonify({'error': 'No selected file'}), 400

# #     try:
# #         img = Image.open(file.stream).convert('RGB')
# #     except UnidentifiedImageError:
# #         return jsonify({'error': 'Cannot identify image file'}), 400

# #     # 원본 이미지에서 얼굴 탐지
# #     boxes, _ = mtcnn.detect(img)
# #     if boxes is None:
# #         return jsonify({'error': 'No face detected'}), 400

# #     # 얼굴 영역만을 예측하여 결과 반환
# #     results = []
# #     for box in boxes:
# #         face = img.crop((box[0], box[1], box[2], box[3]))  # 얼굴 영역 추출
# #         predictions = predict_emotion(face)  # 감정 예측
# #         results.append({
# #             'predictions': [{'label': label, 'probability': float(prob)} for prob, label in predictions]
# #         })

# #     return jsonify({'results': results})

# # if __name__ == '__main__':
# #     app.run(debug=True)


# import io
# import requests  # S3에서 파일을 다운로드하기 위해 사용
# from flask import Flask, request, jsonify
# import torch
# from torchvision import transforms
# from PIL import Image, UnidentifiedImageError
# from flask_cors import CORS
# from facenet_pytorch import MTCNN
# from torchvision import models
# import torch.nn as nn
# import os

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

# device = torch.device("cpu")

# # S3에 있는 pth 파일 URL
# S3_URL = "https://looknlearnmodel.s3.ap-northeast-2.amazonaws.com/best_checkpoint_mtcnn.pth"  # <your-bucket-name>을 실제 버킷 이름으로 변경

# # S3에서 pth 파일 다운로드 함수
# def download_model():
#     response = requests.get(S3_URL)
#     if response.status_code == 200:
#         with open("best_checkpoint_mtcnn.pth", "wb") as f:
#             f.write(response.content)
#         print("Model downloaded successfully from S3")
#     else:
#         print("Failed to download model from S3")

# # 앱 시작 시 모델 다운로드
# download_model()

# # MTCNN 얼굴 탐지 모델 초기화
# mtcnn = MTCNN(keep_all=True, device=device)

# # EfficientNet-B0 모델 불러오기 및 수정
# model_ft = models.efficientnet_b0()
# num_ftrs = model_ft.classifier[1].in_features
# model_ft.classifier = nn.Sequential(
#     nn.Dropout(p=0.5),
#     nn.Linear(num_ftrs, 7)
# )

# # 다운로드한 모델 가중치 로드
# checkpoint_path = './best_checkpoint_mtcnn.pth'
# if os.path.exists(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model_ft.load_state_dict(checkpoint['model_state_dict'], strict=False)
#     print("Model weights loaded successfully")
# else:
#     print("Checkpoint path does not exist")

# model_ft = model_ft.to(device)
# model_ft.eval()

# # 감정 분류 카테고리
# CATEGORIES = ['angry', 'anxiety', 'embarrass', 'hurt', 'joy', 'neutral', 'sad']

# # 이미지 전처리 함수 (224x224로 리사이즈)
# def transform_image(img):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ])
#     return transform(img).unsqueeze(0).to(device)

# # 감정 예측 함수
# def predict_emotion(face_img):
#     face_tensor = transform_image(face_img)
#     with torch.no_grad():
#         outputs = model_ft(face_tensor)
#         probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
    
#     # 확률에 따라 예측 정렬
#     predictions = sorted(zip(probabilities, CATEGORIES), reverse=True, key=lambda x: x[0])
#     return predictions

# # 파일 업로드 엔드포인트
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image part in the request'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     try:
#         img = Image.open(file.stream).convert('RGB')
#     except UnidentifiedImageError:
#         return jsonify({'error': 'Cannot identify image file'}), 400

#     # 원본 이미지에서 얼굴 탐지
#     boxes, _ = mtcnn.detect(img)
#     if boxes is None:
#         return jsonify({'error': 'No face detected'}), 400

#     # 얼굴 영역만을 예측하여 결과 반환
#     results = []
#     for box in boxes:
#         face = img.crop((box[0], box[1], box[2], box[3]))  # 얼굴 영역 추출
#         predictions = predict_emotion(face)  # 감정 예측
#         results.append({
#             'predictions': [{'label': label, 'probability': float(prob)} for prob, label in predictions]
#         })

#     return jsonify({'results': results})

# # if __name__ == '__main__':
# #     app.run(debug=True)

# # Flask 애플리케이션 실행
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)

import io
import requests
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

device = torch.device("cpu")

# S3에 있는 모델 URL
# S3_URL = "https://looknlearnmodel.s3.ap-northeast-2.amazonaws.com/new_best_checkpoint_mtcnn_kdef.pth"
S3_URL = "https://looknlearnmodel.s3.ap-northeast-2.amazonaws.com/best_checkpoint_mtcnn_kdef_93.pth"

# S3에서 모델 다운로드 함수
def download_model():
    local_model_path = "./new_best_checkpoint_mtcnn_kdef.pth"
    if not os.path.exists(local_model_path):  # 모델 파일이 없는 경우
        print("Downloading model from S3...")
        response = requests.get(S3_URL, stream=True)
        if response.status_code == 200:
            with open(local_model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded successfully.")
        else:
            print(f"Failed to download model: HTTP {response.status_code}")
            raise Exception(f"Model download failed with status code {response.status_code}")
    else:
        print("Model already exists locally.")
    return local_model_path

# 앱 시작 시 모델 다운로드
model_path = download_model()

# MTCNN 얼굴 탐지 모델 초기화
mtcnn = MTCNN(keep_all=True, device=device)

# EfficientNet-B0 모델 구조 정의 (weights 없음)
model_ft = models.efficientnet_b0()
num_ftrs = model_ft.classifier[1].in_features
model_ft.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, 7)  # 학습된 모델의 출력 클래스에 맞게 수정
)

# 다운로드한 모델 가중치 로드
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model_ft.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("Model weights loaded successfully.")
else:
    print("Checkpoint path does not exist.")
    raise FileNotFoundError(f"Model file not found: {model_path}")

model_ft = model_ft.to(device)
model_ft.eval()

# 감정 분류 카테고리
CATEGORIES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 이미지 전처리 함수
def transform_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return transform(img).unsqueeze(0).to(device)

# 감정 예측 함수
def predict_emotion(face_img):
    face_tensor = transform_image(face_img)
    with torch.no_grad():
        outputs = model_ft(face_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0).cpu().numpy()
    
    # 확률에 따라 예측 정렬
    predictions = sorted(zip(probabilities, CATEGORIES), reverse=True, key=lambda x: x[0])
    return predictions

# 파일 업로드 엔드포인트
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

    # 원본 이미지에서 얼굴 탐지
    boxes, _ = mtcnn.detect(img)
    if boxes is None:
        return jsonify({'error': 'No face detected'}), 400

    # 얼굴 영역만을 예측하여 결과 반환
    results = []
    for box in boxes:
        face = img.crop((box[0], box[1], box[2], box[3]))  # 얼굴 영역 추출
        predictions = predict_emotion(face)  # 감정 예측
        results.append({
            'predictions': [{'label': label, 'probability': float(prob)} for prob, label in predictions]
        })

    return jsonify({'results': results})

# Flask 애플리케이션 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
