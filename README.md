# 📚 Chat com Múltiplos PDFs (LangChain + Streamlit)

Aplicação web que permite fazer upload de **múltiplos arquivos PDF** e conversar com eles em linguagem natural, usando técnicas de **RAG (Retrieval-Augmented Generation)**. O conteúdo dos documentos é extraído, dividido em chunks, transformado em embeddings e armazenado em um banco vetorial para permitir respostas contextualizadas baseadas no conteúdo dos arquivos.

> Projeto baseado no tutorial [Chat with Multiple PDFs | LangChain App Tutorial in Python](https://www.youtube.com/watch?v=dXxQ0LR-3Hg), com adaptações próprias.

## ✨ Funcionalidades

- 📂 Upload de múltiplos arquivos PDF simultaneamente
- ✂️ Extração e divisão automática do texto em chunks (text splitting)
- 🧠 Geração de embeddings (OpenAI e/ou HuggingFace)
- 🔍 Armazenamento e busca vetorial com FAISS
- 💬 Interface de chat interativa via Streamlit
- 🔁 Memória de conversa (histórico de perguntas e respostas)
- 🔄 Suporte a modelos LLM gratuitos (HuggingFace) e/ou OpenAI (ChatGPT)

## 🛠️ Tecnologias utilizadas

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/) — interface web
- [LangChain](https://www.langchain.com/) — orquestração da pipeline de RAG
- [FAISS](https://github.com/facebookresearch/faiss) — banco de dados vetorial
- [PyPDF2](https://pypi.org/project/PyPDF2/) — extração de texto dos PDFs
- [OpenAI API](https://platform.openai.com/) e/ou [HuggingFace Hub](https://huggingface.co/) — embeddings e modelos de linguagem

## 📋 Pré-requisitos

- Python 3.10+ instalado
- Conta e chave de API da [OpenAI](https://platform.openai.com/api-keys) (opcional, caso use modelos pagos)
- Token de acesso do [HuggingFace](https://huggingface.co/settings/tokens) (opcional, caso use modelos gratuitos)

## 🚀 Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Crie um arquivo `.env` na raiz do projeto com suas chaves de API:
   ```env
   OPENAI_API_KEY=sua_chave_aqui
   HUGGINGFACEHUB_API_TOKEN=seu_token_aqui
   ```

## ▶️ Como executar

```bash
streamlit run app.py
```

Acesse o endereço exibido no terminal (geralmente `http://localhost:8501`) e:

1. Faça upload de um ou mais arquivos PDF na barra lateral
2. Clique em **Process** para processar os documentos
3. Digite suas perguntas no campo de chat e receba respostas baseadas no conteúdo dos PDFs

## 📁 Estrutura do projeto

```
.
├── app.py                 # Aplicação principal (Streamlit)
├── htmlTemplates.py        # Templates de estilo do chat
├── requirements.txt        # Dependências do projeto
├── .env                     # Variáveis de ambiente (não versionado)
└── README.md
```

## 🧩 Como funciona (visão geral)

1. **Extração de texto**: o conteúdo dos PDFs é extraído com `PyPDF2`
2. **Chunking**: o texto é dividido em pedaços menores com `CharacterTextSplitter` do LangChain
3. **Embeddings**: cada chunk é transformado em um vetor numérico (OpenAI ou HuggingFace)
4. **Vector Store**: os vetores são armazenados no FAISS para busca por similaridade
5. **Conversational Chain**: ao fazer uma pergunta, os chunks mais relevantes são recuperados e enviados ao LLM junto com o histórico da conversa para gerar a resposta