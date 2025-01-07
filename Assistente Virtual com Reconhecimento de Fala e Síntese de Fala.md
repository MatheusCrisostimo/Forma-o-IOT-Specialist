# Assistente Virtual com Reconhecimento de Fala e Síntese de Fala

Este projeto implementa um assistente virtual que utiliza **Processamento de Linguagem Natural (PLN)** com reconhecimento de fala (Speech-to-Text) e síntese de fala (Text-to-Speech) para realizar diversas tarefas automatizadas. O sistema é capaz de reconhecer comandos de voz para pesquisar no Wikipedia, abrir vídeos no YouTube, contar piadas, verificar a hora, tocar música, e mais.

## Funcionalidades

- **Reconhecimento de Fala (Speech-to-Text)**: O assistente converte comandos de voz em texto.
- **Síntese de Fala (Text-to-Speech)**: O assistente responde aos comandos com áudio, usando tecnologia de síntese de fala.
- **Pesquisa no Wikipedia**: O assistente pode procurar informações no Wikipedia e ler os resultados.
- **Reprodução de Música**: O assistente pode tocar músicas a partir de um diretório especificado.
- **Abertura de YouTube**: O assistente pode pesquisar no YouTube e abrir vídeos relacionados.
- **Piadas**: O assistente pode contar piadas usando a biblioteca `pyjokes`.
- **Verificação de Hora**: O assistente pode dizer a hora atual.
- **Limpeza da Lixeira**: O assistente pode esvaziar a lixeira do sistema (disponível para Windows).
- **Interação com o Usuário**: O assistente aguarda comandos de voz em loop contínuo.

## Pré-requisitos

Para rodar este projeto, você precisará das seguintes bibliotecas Python:

- **SpeechRecognition**: Para conversão de fala em texto.
- **pyttsx3** ou **gTTS**: Para conversão de texto em fala.
- **wikipedia**: Para buscar informações no Wikipedia.
- **pyjokes**: Para contar piadas.
- **pygame**: Para reprodução de música.
- **pyaudio**: Para captura de áudio do microfone.
- **winshell** (opcional): Para manipulação da lixeira no Windows.
- **webbrowser**: Para abrir o navegador e realizar pesquisas no YouTube.

Você pode instalar todas as dependências necessárias com o seguinte comando:

```bash
pip install SpeechRecognition gTTS pyjokes wikipedia pygame pyaudio winshell
```
---

# Como Usar
## 1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/assistente-virtual.git
cd assistente-virtual
```

## 2. Instale as dependências:

```bash
pip install -r requirements.txt
```

## 3. Execute o script principal:

```bash
python assistente_virtual.py
```

## 4. O assistente começará a escutar comandos de voz e responder com áudio.

#### Exemplos de Comandos

- "Pesquisa [termo] no Wikipedia"
- "Abrir YouTube e procurar por [termo]"
- "Me conte uma piada"
- "Qual é a hora?"
- "Tocar música"
- "Esvaziar lixeira"
- "Sair"

# Estrutura do Projeto
```arduino
assistente-virtual/
│
├── assistente_virtual.py      # Arquivo principal do assistente virtual
├── requirements.txt           # Dependências do projeto
├── README.md                  # Documentação do projeto
└── music/                     # Diretório para suas músicas (se necessário)
```

# Como Funciona
## 1. Reconhecimento de Fala (Speech-to-Text): O assistente usa a biblioteca SpeechRecognition para ouvir o comando de voz e converter em texto.
## 2. Síntese de Fala (Text-to-Speech): Após reconhecer o comando, o assistente utiliza o gTTS ou pyttsx3 para responder com áudio. O áudio é gerado e reproduzido usando a biblioteca playsound ou pygame.mixer (dependendo da plataforma).
## 3. Ações do Assistente: O assistente possui diversas funções, como pesquisa no Wikipedia, abertura de links no YouTube, contagem de piadas, e execução de comandos no sistema, como limpar a lixeira.
## 4. Interação em Loop: O assistente ficará ouvindo e respondendo a comandos em um loop contínuo até que o comando "sair" seja reconhecido.

# Como Contribuir
## 1. Faça o fork deste repositório.
## 2. Crie uma branch para a sua feature (git checkout -b minha-feature).
## 3. Faça suas alterações e commit (git commit -am 'Adicionando nova funcionalidade').
## 4. Push para a branch (git push origin minha-feature).
## 5. Abra um pull request para o repositório original.

# Licença
Este projeto está licenciado sob a MIT License - veja o arquivo LICENSE para mais detalhes.

# Agradecimentos
- gTTS - Para conversão de texto em fala.
- SpeechRecognition - Para reconhecimento de fala.
- Wikipedia API - Para busca de artigos no Wikipedia.
- PyJokes - Para contar piadas.
- pygame - Para reprodução de música.
