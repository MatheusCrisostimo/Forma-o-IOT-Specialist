# Cálculo de Métricas de Avaliação de Aprendizado

Neste projeto, vamos calcular as principais métricas para avaliação de modelos de classificação de dados, como acurácia, sensibilidade (recall), especificidade, precisão e F-score. Para que seja possível implementar estas funções, você deve utilizar os métodos e suas fórmulas correspondentes (Tabela 1).

Para a leitura dos valores de VP, VN, FP e FN, será necessário escolher uma matriz de confusão para a base dos cálculos. Essa matriz você pode escolher de forma arbitrária, pois nosso objetivo é entender como funciona cada métrica.

Tabela 1: Visão geral das métricas usadas para avaliar métodos de classificação. VP: verdadeiros positivos; FN: falsos negativos; FP: falsos positivos; VN: verdadeiros negativos; P: precisão; S: sensibilidade; N: total de elementos.

## 🚀 Funcionalidades
- **Cálculo de Acurácia**
- **Cálculo de Sensibilidade (Recall)**
- **Cálculo de Especificidade**
- **Cálculo de Precisão**
- **Cálculo de F-score**

## 🛠️ Tecnologias Utilizadas
- Python 3.x

## 🧰 Pré-requisitos
Certifique-se de ter o Python instalado em sua máquina.

## 📂 Estrutura do Projeto
```plaintext
📦 calculo-metricas-avaliacao
├── confusion_matrix.py    # Código Python para o cálculo das métricas
└── README.md              # Descrição do projeto
```

## 📝 Como Usar
1. Clone o Repositório:

```bash
git clone https://github.com/SEU_USUARIO/calculo-metricas-avaliacao.git
cd calculo-metricas-avaliacao
```

2. Execute o Script: Execute o arquivo Python para calcular as métricas:

```bash
python confusion_matrix.py
```

### Saída Esperada
O script exibirá os valores calculados para acurácia, sensibilidade, especificidade, precisão e F-score.

## 📖 Código Explicado
O código principal para o cálculo das métricas é:

```python
def calcular_metricas(VP, VN, FP, FN):
    acuracia = (VP + VN) / (VP + VN + FP + FN)
    sensibilidade = VP / (VP + FN)
    especificidade = VN / (VN + FP)
    precisao = VP / (VP + FP)
    f_score = 2 * (precisao * sensibilidade) / (precisao + sensibilidade)
    
    return acuracia, sensibilidade, especificidade, precisao, f_score

# Exemplo de uso
VP = 50
VN = 40
FP = 10
FN = 5

metricas = calcular_metricas(VP, VN, FP, FN)
print(f"Acurácia: {metricas[0]:.2f}")
print(f"Sensibilidade: {metricas[1]:.2f}")
print(f"Especificidade: {metricas[2]:.2f}")
print(f"Precisão: {metricas[3]:.2f}")
print(f"F-score: {metricas[4]:.2f}")
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
