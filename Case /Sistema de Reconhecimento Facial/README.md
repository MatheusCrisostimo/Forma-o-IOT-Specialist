# Sistema de Reconhecimento Facial com TensorFlow

Este projeto é um sistema de reconhecimento facial que utiliza TensorFlow para detectar e reconhecer rostos em tempo real.

## Estrutura do Projeto

- `main.py`: Ponto de entrada do sistema.
- `src/face_detection.py`: Código relacionado à detecção de faces.
- `src/face_recognition.py`: Código para reconhecimento facial.
- `src/camera.py`: Código para captura de imagens.
- `models/facenet_model.h5`: Modelo pré-treinado FaceNet.
- `images/reference_image.jpg`: Imagem de referência para reconhecimento.

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu_usuario/seu_repositorio.git
   cd seu_repositorio

!pip install tensorflow opencv-python-headless keras

import tensorflow as tf
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files
from PIL import Image

# Carregue o classificador Haar Cascade para detecção de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Função para detectar faces em uma imagem
def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

# Função para reconhecer o rosto comparando com a face de referência
def recognize_face(face, model, known_face_encoding):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    face = np.expand_dims(face, axis=0)
    encoding = model.predict(face)
    distance = np.linalg.norm(encoding - known_face_encoding)
    return distance

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Erro ao capturar a imagem.")
        return None
    return frame

# Faça o upload do arquivo do modelo FaceNet (.h5)
uploaded = files.upload()

# Pegue o nome do arquivo carregado
model_filename = list(uploaded.keys())[0]
print(f'Arquivo carregado: {model_filename}')

# Verifique se o arquivo é um .h5
if model_filename.endswith('.h5'):
    try:
        # Carregue o modelo usando o nome correto
        model = tf.keras.models.load_model(model_filename)
        print("Modelo carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
else:
    print("Erro: O arquivo carregado não é um modelo .h5.")

# Continuar apenas se o modelo foi carregado com sucesso
if 'model' in locals():
    # Carregar imagem de referência
    uploaded_files = files.upload()
    reference_img_filename = list(uploaded_files.keys())[0]
    reference_img = cv2.imread(reference_img_filename)
    
    if reference_img is None:
        print("Erro ao carregar a imagem de referência.")
    else:
        reference_faces = detect_faces(reference_img)
        if len(reference_faces) > 0:
            x, y, w, h = reference_faces[0]
            reference_face = reference_img[y:y+h, x:x+w]
            known_face_encoding = recognize_face(reference_face, model, np.zeros((1, 128)))  # Iniciando com um vetor zero para passar a verificação

            # Capturar imagem da câmera
            camera_img = capture_image()
            if camera_img is not None:
                camera_faces = detect_faces(camera_img)
                for (x, y, w, h) in camera_faces:
                    face = camera_img[y:y+h, x:x+w]
                    distance = recognize_face(face, model, known_face_encoding)
                    label = "Match" if distance < 0.6 else "No Match"
                    cv2.rectangle(camera_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(camera_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2_imshow(camera_img)
            else:
                print("Erro ao capturar imagem da câmera.")
        else:
            print("Nenhuma face detectada na imagem de referência.")
else:
    print("O modelo não foi carregado. Verifique o arquivo e tente novamente.")
