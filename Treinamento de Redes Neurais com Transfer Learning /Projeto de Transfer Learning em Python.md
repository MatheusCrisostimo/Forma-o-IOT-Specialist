# Estrutura do Repositório no GitHub
```
Transfer-Learning-Project/
│
├── README.md
├── dataset/
│   └── [dataset personalizado ou link para download do Cats vs Dogs]
├── notebook/
│   └── transfer_learning_cats_vs_dogs.ipynb
├── images/
│   └── [imagens para ilustração, gráficos, etc.]
└── LICENSE
```

# Transfer Learning com TensorFlow e VGG16

Este projeto demonstra como realizar **Transfer Learning** para classificação de imagens utilizando a rede pré-treinada **VGG16** no TensorFlow/Keras. O objetivo é classificar imagens de **gatos** e **cachorros** usando o dataset *Cats vs Dogs*, mas você pode adaptar o projeto para outras classes de sua escolha.

---

## Estrutura do Projeto

### 1. Dataset
- Dataset utilizado: [Microsoft Cats vs Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765).
- **Estrutura esperada do dataset**:
  - A pasta `dataset` deve conter subpastas para cada classe:
    ```
    dataset/
    ├── Cat/
    │   ├── cat1.jpg
    │   ├── cat2.jpg
    │   └── ...
    └── Dog/
        ├── dog1.jpg
        ├── dog2.jpg
        └── ...
    ```

### 2. Transfer Learning
A técnica de Transfer Learning consiste em:
1. Utilizar o modelo **VGG16** pré-treinado no ImageNet.
2. Remover a última camada do modelo.
3. Adicionar camadas personalizadas para classificar imagens em **2 classes** (gatos e cachorros).
4. Treinar somente as novas camadas, congelando as anteriores.

---

## Requisitos

Para executar o projeto, você precisará:
- Python 3.7 ou superior
- TensorFlow 2.x
- Matplotlib
- Google Colab (opcional)

Instale os pacotes necessários com:

```bash
pip install tensorflow matplotlib
```

# Como Executar
## 1. Baixe o dataset:
Acesse Cats vs Dogs Dataset.
Extraia os arquivos na pasta dataset.

## 2. Execute o notebook:

Abra o notebook transfer_learning_cats_vs_dogs.ipynb no Google Colab.
Siga os passos descritos para treinar o modelo e avaliar os resultados.

## 3. Personalize o dataset:

Para treinar com suas próprias imagens, organize-as como no exemplo da estrutura do dataset.

# Resultados
Gráficos de treinamento e validação:

- Perda de Validação
- Acurácia de Validação

Exemplo de previsão em imagem nova:

Entrada:
Saída: Classe prevista: Gato

# Licença
Este projeto é licenciado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

---
### Código Python no Notebook (transfer_learning_cats_vs_dogs.ipynb)

Aqui está o código completo que deve ser usado no notebook Colab.
```python
# Transfer Learning com VGG16 - Classificação de Gatos e Cachorros

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D

# Configuração do ambiente
dataset_dir = './cats_vs_dogs/PetImages'
batch_size = 32
img_height, img_width = 224, 224

# Preprocessamento dos dados
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,  # 20% para validação
    horizontal_flip=True,
    zoom_range=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Carregar o modelo base (VGG16 pré-treinado)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Congelar camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Adicionar camadas personalizadas
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Saída binária (0 ou 1)
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Avaliação
loss, accuracy = model.evaluate(validation_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Gráficos
fig, axs = plt.subplots(1, 2, figsize=(16, 4))

# Perda
axs[0].plot(history.history["loss"], label="Train Loss")
axs[0].plot(history.history["val_loss"], label="Validation Loss")
axs[0].set_title("Loss")
axs[0].set_xlabel("Epochs")
axs[0].legend()

# Acurácia
axs[1].plot(history.history["accuracy"], label="Train Accuracy")
axs[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
axs[1].set_title("Accuracy")
axs[1].set_xlabel("Epochs")
axs[1].legend()

plt.show()

# Previsão em uma imagem nova
def predict_image(path, model):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicted Class: {'Dog' if prediction[0] > 0.5 else 'Cat'}")
    plt.show()

# Teste
predict_image('./cats_vs_dogs/PetImages/Cat/1.jpg', model)
```

# Publicando no GitHub
## 1. Crie um repositório no GitHub, por exemplo, ``Transfer-Learning-Project``.
## 2. Clone o repositório em sua máquina:
```
git clone https://github.com/seu-usuario/Transfer-Learning-Project.git
```

## 3. Copie os arquivos estruturados (README.md, notebook, dataset, etc.) para a pasta do repositório.
## 4. Adicione e envie os arquivos:
```
git add .
git commit -m "Initial commit for Transfer Learning Project"
git push origin main
```
