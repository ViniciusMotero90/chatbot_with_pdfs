import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from htmlTemplate import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    txt = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            txt += page.extract_text()
    return txt

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    retriever = vectorstore.as_retriever()

    # Prompt para reformular a pergunta com base no histórico
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Dado o histórico da conversa e a pergunta do usuário, reformule a "
                   "pergunta para que seja autossuficiente. Não responda, apenas "
                   "reformule (ou retorne igual, se já estiver clara)."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    contextualize_chain = contextualize_prompt | llm | StrOutputParser()

    def route(input_dict):
        if input_dict.get("chat_history"):
            return contextualize_chain
        return input_dict["input"]

    history_aware_retriever = (
        RunnablePassthrough.assign(standalone_question=route)
        | (lambda x: retriever.invoke(x["standalone_question"]))
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Você é um assistente que responde perguntas sobre os documentos "
                   "do usuário. Use o contexto abaixo para responder. Se não souber, "
                   "diga que não sabe.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    rag_chain = (
        RunnablePassthrough.assign(context=history_aware_retriever | format_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # Memória em sessão (armazenada no session_state pra persistir entre reruns do Streamlit)
    if "store" not in st.session_state:
        st.session_state.store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversation_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation.invoke(
        {"input": user_question},
        config={"configurable": {"session_id": "default"}},
    )
    history = st.session_state.store["default"].messages
    for i, message in enumerate(history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Documentos processados com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao processar documentos: {e}")

if __name__ == "__main__":
    main()