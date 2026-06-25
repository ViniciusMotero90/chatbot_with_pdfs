import streamlit as st
import uuid
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import warnings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from htmlTemplate import css, bot_template, user_template
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

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
    
    def call_model(state: MessagesState):
        print("DEBUG state recebido:", state)
        question = state["messages"][-1].content
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        
        prompt = f"Use o contexto abaixo para responder. \n\nContexto:\n{context}\n\nPergunta:{question}"
        response = llm.invoke([{"role":"user","content":prompt}])
        return {"messages": [response]}
    workflow = StateGraph(MessagesState)
    workflow.add_node("model",call_model)
    workflow.add_edge(START,"model")
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

def handle_userinput(user_question):
    config = {"configurable": {"thread_id": st.session_state.session_id}}
    response = st.session_state.conversation.invoke(
        {"messages": [{"role": "user", "content": user_question}]},
        config=config,
    )
    for i, message in enumerate(response["messages"]):
        if message.type == "human":
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
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

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