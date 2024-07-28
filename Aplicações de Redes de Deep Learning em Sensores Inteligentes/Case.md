# Projeto de Reconhecimento de Imagens com Transfer Learning

## Descrição

Este projeto implementa uma rede neural para reconhecimento de imagens digitais utilizando Transfer Learning com o modelo InceptionV3.

## Estrutura do Projeto

- `train.py`: Script principal para treinamento e avaliação do modelo.
- `data/`: Diretório contendo os dados de treinamento e validação.
- `requirements.txt`: Arquivo com as dependências do projeto.

## Como Usar

### 1. Instale as dependências

Primeiro, crie um arquivo `requirements.txt` com as bibliotecas necessárias:

```sh
echo "tensorflow" >> requirements.txt
echo "numpy" >> requirements.txt
echo "matplotlib" >> requirements.txt
echo "pandas" >> requirements.txt
echo "scikit-learn" >> requirements.txt
```

Então, instale as dependências com:
pip install -r requirements.txt

### 2. Organize seus dados
Crie a estrutura de diretórios para seus dados de treinamento e validação:

````sh
mkdir -p data/train data/validation
````

Coloque suas imagens nas pastas `data/train`e `data/validation`conforme necessário. Cada classe deve ter seu próprio subdiretório.

### 3. Defina e Treine o Modelo

``` python
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Carregar o modelo pré-treinado InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Adicionar novas camadas para adaptação à tarefa específica
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)  # Ajuste o número de saídas conforme necessário

model = Model(inputs=base_model.input, outputs=predictions)

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Preparar os dados com ImageDataGenerator para data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Treinar o modelo
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10  # Ajuste o número de épocas conforme necessário
)

# Avaliar o modelo
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Accuracy: {accuracy*100:.2f}%')

# Salvar o modelo treinado
model.save('model.h5')
```
Para treinar o modelo, execute:

``` sh
python train.py
```

### 4. Avalie o Modelo
Os resultados da avaliação serão exibidos após o treinamento.

### 5. Suba o Projeto no GitHub

#### 1. Inicialize um novo repositório Git:
``` sh
git init
```

#### 2. Adicione todos os arquivos ao repositório:
``` sh
git add .
```

#### 3. Faça o commit das mudanças:
``` sh
git commit -m "Initial commit with project structure and training script"
```

#### 4. Adicione a URL do repositório remoto:
``` sh
git remote add origin https://github.com/seu-usuario/image-recognition-deep-learning.git
```

#### 5. Envie as mudanças para o GitHub:
``` sh
git push -u origin master
```

## Estrutura Final do Projeto
Seu projeto deve ter a seguinte estrutura:

``` arduino
image-recognition-deep-learning/
│
├── data/
│   ├── train/
│   │   ├── class1/
│   │   └── class2/
│   └── validation/
│       ├── class1/
│       └── class2/
│
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

## .gitignore
Crie um arquivo .gitignore para evitar que arquivos desnecessários sejam enviados ao repositório:

``` kotlin
data/
model.h5
__pycache__/
*.pyc
.DS_Store
```

## Transfer Learning
O modelo base utilizado é o InceptionV3, pré-treinado no ImageNet, com novas camadas adicionadas para adaptação à tarefa específica de classificação.

## Licença
Este projeto está licenciado sob a MIT License.

``` csharp

### 6. Atualize o Repositório

Caso faça alterações, atualize o repositório com as últimas mudanças:
```

```sh
git add .
git commit -m "Updated with detailed instructions and Transfer Learning"
git push
```
