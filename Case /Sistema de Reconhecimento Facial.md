### Sistema de Reconhecimento Facial com TensorFlow no Google Colab

Para criar um sistema de reconhecimento facial que detecta e reconhece faces usando TensorFlow, siga os passos abaixo:

#### 1. Configuração do Ambiente
Primeiro, configure o ambiente no Google Colab:

```python
!pip install tensorflow opencv-python-headless
```

#### 2. Importação das Bibliotecas
Importe as bibliotecas necessárias:

```python
import tensorflow as tf
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
```

#### 3. Carregamento do Modelo de Detecção de Faces
Utilize um modelo pré-treinado para detecção de faces, como o `Haar Cascade`:

```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
```

#### 4. Função para Detecção de Faces
Crie uma função para detectar faces em uma imagem:

```python
def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces
```

#### 5. Carregamento do Modelo de Reconhecimento Facial
Utilize um modelo de reconhecimento facial, como o `FaceNet`:

```python
model = tf.keras.models.load_model('path_to_facenet_model')
```

#### 6. Função para Reconhecimento Facial
Crie uma função para reconhecer a face:

```python
def recognize_face(face, model, known_face_encoding):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    face = np.expand_dims(face, axis=0)
    encoding = model.predict(face)
    distance = np.linalg.norm(encoding - known_face_encoding)
    return distance
```

#### 7. Captura de Imagem da Câmera
Capture uma imagem da câmera:

```python
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame
```

#### 8. Pipeline Completo
Combine todas as funções para criar o pipeline completo:

```python
# Carregar imagem de referência
reference_img = cv2.imread('path_to_reference_image')
reference_faces = detect_faces(reference_img)
x, y, w, h = reference_faces[0]
reference_face = reference_img[y:y+h, x:x+w]
known_face_encoding = recognize_face(reference_face, model)

# Capturar imagem da câmera
camera_img = capture_image()
camera_faces = detect_faces(camera_img)

for (x, y, w, h) in camera_faces:
    face = camera_img[y:y+h, x:x+w]
    distance = recognize_face(face, model, known_face_encoding)
    if distance < 0.6:
        label = "Match"
    else:
        label = "No Match"
    cv2.rectangle(camera_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(camera_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

cv2_imshow(camera_img)
```

### Explicação
- **Detecção de Faces**: Utiliza `Haar Cascade` para detectar faces na imagem.
- **Reconhecimento Facial**: Utiliza `FaceNet` para comparar a face detectada com a face de referência.
- **Captura de Imagem**: Captura uma imagem da câmera para reconhecimento.

Este código permite que o usuário insira uma foto e utilize a câmera para verificar se a pessoa na imagem é a mesma capturada pela câmera.
