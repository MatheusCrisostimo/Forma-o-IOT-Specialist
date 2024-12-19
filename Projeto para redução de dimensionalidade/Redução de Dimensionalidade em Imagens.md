# 🖼️ Redução de Dimensionalidade em Imagens

Este projeto implementa a transformação de uma imagem colorida para tons de cinza e uma versão binarizada (preto e branco) usando Python. O objetivo é demonstrar técnicas de pré-processamento de imagens para aplicações em redes neurais e aprendizado de máquina.

## 🚀 Funcionalidades
- **Conversão para Tons de Cinza**: Reduz a imagem para uma escala de intensidade de 0 a 255.
- **Binarização (Preto e Branco)**: Converte a imagem para dois valores possíveis (0 ou 255), com base em um limiar.

## 🛠️ Tecnologias Utilizadas
- Python 3.x
- OpenCV - Biblioteca para processamento de imagens.
- Matplotlib - Visualização gráfica.
- NumPy - Operações matemáticas.

## 🧰 Pré-requisitos
Certifique-se de ter o Python instalado em sua máquina. Depois, instale as dependências necessárias com:

```bash
pip install opencv-python matplotlib numpy
```

## 📂 Estrutura do Projeto
```plaintext
📦 reducao-dimensionalidade-imagens
├── lena.jpg               # Imagem de entrada (Lena)
├── transform_imagem.py    # Código Python para o processamento
└── README.md              # Descrição do projeto
```

## 📝 Como Usar
1. Clone o Repositório:

```bash
git clone https://github.com/SEU_USUARIO/reducao-dimensionalidade-imagens.git
cd reducao-dimensionalidade-imagens
```

2. Adicione a Imagem de Entrada: Certifique-se de que a imagem `lena.jpg` está no mesmo diretório do script. Se desejar usar outra imagem, substitua o nome no código.

3. Execute o Script: Execute o arquivo Python para processar a imagem:

```bash
python transform_imagem.py
```

### Saída Esperada
O script exibirá três imagens:
- Imagem Original (Colorida)
- Imagem em Tons de Cinza
- Imagem Binarizada (Preto e Branco)

## ✨ Resultados
### Entrada
Imagem colorida usada como entrada:

### Saída
#### Imagem em Tons de Cinza:
Mostra a intensidade de luz de cada pixel no intervalo [0, 255].

#### Imagem Binarizada (Preto e Branco):
Apenas dois valores possíveis para cada pixel: 0 ou 255.

## 📖 Código Explicado
O código principal para a transformação é:

```python
import cv2
import matplotlib.pyplot as plt

# Carregar a imagem colorida
imagem_colorida = cv2.imread("lena.jpg")

# Converter a imagem de BGR (OpenCV) para RGB
imagem_rgb = cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2RGB)

# Converter para tons de cinza
imagem_cinza = cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2GRAY)

# Binarizar a imagem
_, imagem_binarizada = cv2.threshold(imagem_cinza, 127, 255, cv2.THRESH_BINARY)

# Visualizar os resultados
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(imagem_rgb)
ax[0].set_title("Imagem Colorida")
ax[1].imshow(imagem_cinza, cmap="gray")
ax[1].set_title("Tons de Cinza")
ax[2].imshow(imagem_binarizada, cmap="gray")
ax[2].set_title("Binarizada")
plt.show()
```

## 🤝 Contribuições
Contribuições são bem-vindas! Siga os passos para criar um pull request:

1. Faça um fork do projeto.
2. Crie uma nova branch:
```bash
git checkout -b minha-melhoria
```
3. Faça suas alterações e commit:
```bash
git commit -m "Descrição da melhoria"
```
4. Envie as alterações:
```bash
git push origin minha-melhoria
```
5. Abra um pull request no repositório original.

## 📜 Licença
Este projeto está sob a licença MIT. Consulte o arquivo LICENSE para mais informações.
```

Você pode copiar e colar esse conteúdo no arquivo `README.md` do seu repositório. Se precisar de mais alguma coisa, estou à disposição!
