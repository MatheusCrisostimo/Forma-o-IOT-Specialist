# Automação por Meio de Reconhecimento de Voz

Este projeto visa criar uma assistente virtual para automação residencial utilizando reconhecimento de voz, Arduino, motores, lâmpadas, sirenes e outros elementos de automação.

## Instalação

```bash
pip install SpeechRecognition
pip install pyaudio
```

```bash automacao_voz.py
import speech_recognition as sr
import os

# Função para ouvir e reconhecer a fala
def ouvir_microfone():
    # Habilita o microfone do usuário
    microfone = sr.Recognizer()

    # Usando o microfone
    with sr.Microphone() as source:
        # Chama um algoritmo de redução de ruídos no som
        microfone.adjust_for_ambient_noise(source)
        # Frase para o usuário dizer algo
        print("Diga alguma coisa: ")
        # Armazena o que foi dito numa variável
        audio = microfone.listen(source)

    try:
        # Passa a variável para o algoritmo reconhecedor de padrões
        frase = microfone.recognize_google(audio, language='pt-BR')

        if "navegador" in frase:
            os.system("start Chrome.exe")
        elif "Excel" in frase:
            os.system("start Excel.exe")

        # Retorna a frase pronunciada
        print("Você disse: " + frase)
    except sr.UnknownValueError:
        print("Não entendi")

    return frase

ouvir_microfone()
```

# Adicionar e Comitar o Código:
Adicione o arquivo ao repositório:
git add automacao_voz.py

# Faça um commit das mudanças:
git commit -m "Adiciona código base para automação por voz"

# Enviar para o GitHub:
Envie as mudanças para o repositório remoto:
git push origin main

