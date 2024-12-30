# Sistema de Recomendação por Imagens

Bem-vindo ao repositório do Sistema de Recomendação por Imagens! Este projeto demonstra o uso de Deep Learning para criar um sistema de recomendação baseado na similaridade de imagens. O objetivo é identificar e sugerir produtos visualmente semelhantes com base em características físicas como formato, cor, textura e outros aspectos visuais.
---
## 📂 Estrutura do Projeto
```plaintext
sistema-recomendacao-imagens/
├── data/                # Dataset de imagens para treinamento
├── models/              # Modelos treinados
├── notebooks/           # Notebooks de experimentação
├── src/                 # Código fonte do projeto
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── feature_extraction.py
│   ├── similarity_calculation.py
│   ├── recommendation_system.py
├── app.py               # Aplicação web
├── requirements.txt     # Dependências do projeto
└── README.md            # Documentação do projeto
```
---

## 🎯 Objetivos do Projeto

1. **Treinamento Personalizado**: Treinar o modelo YOLO para detectar pelo menos duas classes customizadas.
2. **Transferência de Aprendizado**: Utilizar pesos pré-treinados para acelerar o treinamento.
3. **Resultados Visuais**: Analisar o desempenho do modelo em imagens de teste.

---

## 🛠️ Ferramentas Utilizadas

- **Python**
- **TensorFlow/Keras**: Para treinamento da rede neural.
- **Streamlit/Flask**: Para desenvolvimento de interface web (opcional).
- **OpenCV**: Para processamento de imagens.
- **Google Colab**: Como ambiente de desenvolvimento e experimentação.
- **Cosine Similarity**: Para medir similaridade entre imagens.

---

## 🚀 Etapas do Desenvolvimento

### 1. Descrição do Desafio

Sistemas de Recomendação são amplamente utilizados para melhorar a experiência do usuário em plataformas digitais. Neste projeto, o desafio é:

- Desenvolver um modelo capaz de classificar imagens por similaridade visual.
- Utilizar este modelo para recomendar produtos relacionados, ignorando dados textuais como preço, marca ou loja.

#### 1.1 Coleta e Pré-processamento de Dados
  ```pyhton
# src/data_preprocessing.py

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_images(image_dir, image_size=(224, 224), augment=False):
    images = []
    labels = []
    
    # Data augmentation
    if augment:
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    
    for label in os.listdir(image_dir):
        class_dir = os.path.join(image_dir, label)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, image_size)
            image = image / 255.0  # Normalização
            
            images.append(image)
            labels.append(label)

            # Augmentação se ativada
            if augment:
                augmented_images = datagen.flow(np.expand_dims(image, axis=0), batch_size=1)
                for i in range(5):  # Gerar 5 imagens aumentadas por imagem
                    augmented_image = augmented_images.next()[0]
                    images.append(augmented_image)
                    labels.append(label)
    
    return np.array(images), np.array(labels)
 ```
#### 1.2 Treinamento do Modelo
  ```pyhton
# src/model_training.py

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def build_model(num_classes, base_model_trainable=False):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = base_model_trainable  # Controlar se as camadas do ResNet50 serão treináveis
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, train_images, train_labels, val_images, val_labels, epochs=10, batch_size=32):
    model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=epochs, batch_size=batch_size)
    return model
  ```
#### 1.3 Extração de Características
  ```pyhton
# src/feature_extraction.py

from tensorflow.keras.models import Model

def extract_features(model, images, layer_name='global_average_pooling2d'):
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    features = feature_extractor.predict(images)
    return features
  ```
#### 1.4 Cálculo de Similaridade
  ```pyhton
# src/similarity_calculation.py

from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(features, query_feature):
    similarities = cosine_similarity(query_feature, features)
    return similarities
  ```

#### 1.5 Coleta e Pré-processamento de Dados
  ```pyhton
# src/recommendation_system.py

import numpy as np
from src.similarity_calculation import calculate_similarity

def recommend_similar_images(features, query_feature, top_k=5):
    similarities = calculate_similarity(features, query_feature)
    similar_indices = np.argsort(similarities, axis=1)[0][-top_k:][::-1]  # Top-K índices
    return similar_indices
  ```

#### 1.6 Aplicação Web
  ```pyhton
# app.py

import streamlit as st
import cv2
import numpy as np
from src.data_preprocessing import load_and_preprocess_images
from src.model_training import build_model, train_model
from src.feature_extraction import extract_features
from src.recommendation_system import recommend_similar_images

# Carregar e pré-processar imagens
image_dir = 'data/'
images, labels = load_and_preprocess_images(image_dir, augment=True)

# Codificar labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Treinar o modelo
num_classes = len(set(labels))
model = build_model(num_classes, base_model_trainable=True)
model = train_model(model, images, labels_encoded, images, labels_encoded, epochs=10, batch_size=32)

# Extrair características
features = extract_features(model, images)

# Interface do usuário
st.title('Sistema de Recomendação por Imagens')

uploaded_file = st.file_uploader("Escolha uma imagem...", type="jpg")

if uploaded_file is not None:
    # Carregar a imagem
    query_image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    query_image = cv2.resize(query_image, (224, 224))
    query_image = query_image / 255.0
    
    # Extrair as características da imagem consultada
    query_feature = extract_features(model, np.array([query_image]))
    
    # Obter imagens semelhantes
    similar_indices = recommend_similar_images(features, query_feature, top_k=5)
    
    st.write("Imagens recomendadas:")
    for idx in similar_indices:
        st.image(images[idx], caption=labels[idx])
  ```
#### 1.7 Coleta e Pré-processamento de Dados
  ```plaintext
tensorflow
streamlit
opencv-python
numpy
scikit-learn
  ```

### 2. Execução
- Clone o repositório:
```
git clone https://github.com/seu-usuario/sistema-recomendacao-imagens.git
cd sistema-recomendacao-imagens
```
- Instale as dependências:
```
pip install -r requirements.txt
```
- Execute a aplicação:
 ```
  streamlit run app.py
 ```


