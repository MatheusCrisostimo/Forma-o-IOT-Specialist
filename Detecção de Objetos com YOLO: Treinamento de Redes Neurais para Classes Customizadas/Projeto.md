# Detecção de Objetos com YOLO: Treinamento de Redes Neurais para Classes Customizadas

Este projeto explora a implementação da rede YOLO (You Only Look Once) para detecção de objetos. Utilizando transferência de aprendizado no Google Colab, configuramos e treinamos o modelo para identificar classes customizadas, além de aproveitar as classes pré-treinadas.

---
## 📂 Estrutura do Projeto
- `data/`: Contém os arquivos de dados e anotações.
- `cfg/`: Contém o arquivo de configuração do YOLO.
- `scripts/`: Scripts para configurar o ambiente, treinar e avaliar o modelo.
- `weights/`: Contém os pesos pré-treinados.
  
## 🎯 Objetivos do Projeto
1. **Treinamento Personalizado**: Treinar o modelo YOLO para detectar pelo menos duas classes customizadas.
2. **Transferência de Aprendizado**: Utilizar pesos pré-treinados para acelerar o treinamento.
3. **Resultados Visuais**: Analisar o desempenho do modelo em imagens de teste.
---

## 🛠️ Ferramentas Utilizadas
- **LabelMe**: Para rotulação manual de imagens ([Link](http://labelme.csail.mit.edu/Release3.0/)).
- **Google Colab**: Para treinamento e execução do modelo em GPUs.
- **Darknet**: Framework para YOLO ([Link](https://pjreddie.com/darknet/yolo/)).
- **COCO Dataset**: Dataset pré-rotulado para detecção de objetos ([Link](https://cocodataset.org/#home)).
---

## 🚀 Etapas do Desenvolvimento
### 1. Rotulação de Imagens
- Use o [LabelMe](http://labelme.csail.mit.edu/Release3.0/) para rotular suas imagens manualmente.
- Organize as imagens e os arquivos de anotação conforme o formato YOLO.

### 2. Configuração do Ambiente
- Monte o Google Drive:
  ```python
  from google.colab import drive
  drive.mount('/content/gdrive')
 
- Clone o repositório Darknet e compile com suporte a GPU e OpenCV:
  ```
  !git clone https://github.com/AlexeyAB/darknet
  %cd darknet
  !sed -i 's/GPU=0/GPU=1/' Makefile
  !sed -i 's/OPENCV=0/OPENCV=1/' Makefile
  !make
### 3. Preparação dos Dados
- Ajuste dos arquivos de configuração do YOLO (`obj.data`, `train.txt`, `test.txt`).
- Criação de um dataset balanceado contendo classes específicas.

- obj.data:
 ```
  classes=2
  train=/content/gdrive/MyDrive/darknet/train.txt
  valid=/content/gdrive/MyDrive/darknet/test.txt
  names=/content/gdrive/MyDrive/darknet/obj.names
  backup=/content/gdrive/MyDrive/darknet/backup/
 ```
- obj.names:
 ```
  dog
  cat
 ```
- Modifique os filtros e as classes conforme o número de categorias:
- yolov3.cfg:
 ```
  [convolutional]
  filters=(classes + 5) * 3
  
  [yolo]
  classes=2
 ```

### 4. Treinamento do Modelo
- Aplicação de transferência de aprendizado utilizando o arquivo de pesos `darknet53.conv.74`.
- Salvamento automático dos pesos treinados durante o processo.

- Baixe os pesos pré-treinados:
```
  !wget https://pjreddie.com/media/files/darknet53.conv.74
 ```
- Inicie o treinamento:
```
    !./darknet detector train \
    /content/gdrive/MyDrive/darknet/obj.data \
    /content/gdrive/MyDrive/darknet/yolov3.cfg \
    /content/gdrive/MyDrive/darknet/darknet53.conv.74 \
    -dont_show
 ```

### 5. Testes e Avaliação
- Testes realizados com imagens específicas para avaliar a precisão e a eficácia do modelo treinado.
- Visualização dos resultados por meio de imagens anotadas.

- Avalie o modelo:
```
    !./darknet detector map \
    /content/gdrive/MyDrive/darknet/obj.data \
    /content/gdrive/MyDrive/darknet/yolov3.cfg \
    /content/gdrive/MyDrive/darknet/backup/yolov3_last.weights
 ```
- Teste o modelo em uma imagem:
```
  !./darknet detector test \
    /content/gdrive/MyDrive/darknet/obj.data \
    /content/gdrive/MyDrive/darknet/yolov3.cfg \
    /content/gdrive/MyDrive/darknet/backup/yolov3_last.weights \
    -ext_output data/test_image.jpg
 ```
- Exiba a imagem com as detecções:
 ``` python 
import cv2
import matplotlib.pyplot as plt

def imShow(path):
    image = cv2.imread(path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

imShow('predictions.jpg')
 ```

---

## 🔎 Resultados
Os resultados incluem imagens com os objetos detectados marcados com caixas delimitadoras. Exemplo:

## 📚 Referências
- YOLO Darknet Documentation
- COCO Dataset
- LabelMe Annotation Tool

## 📝 Licença
Este projeto está licenciado sob a Licença MIT. Veja o arquivo LICENSE para mais detalhes.
