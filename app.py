from importlib.util import resolve_name
from pyexpat import model
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFaceHub
from htmlTemplate import css, bot_template, user_template


def get_pdf_text(pdf_docx):
    text = ""
    for pdf in pdf_docx:
        file_reader = PdfReader(pdf)
        for page in file_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):    
    embedding = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-large')
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embedding)
    return vector_store


def get_conversation_chain(vectorstore):
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    #llm = HuggingFaceHub(repo_id="microsoft/DialoGPT-medium", model_kwargs={"temperature":0.5, "max_length":512})
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.2, "max_length":512})
    #llm = HuggingFaceHub(repo_id="tiiuae/falcon-40b", model_kwargs={"temperature":0.4, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conver_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= vectorstore.as_retriever(),
        memory = memory
    )
    return conver_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()

    st.set_page_config(page_title='Chat with PDFs')
    st.write(css, unsafe_allow_html=True)
    st.title("MA-LTSS QnA Bot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    #st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question here:")
    #st.write(user_question)
    if user_question:
        try:
            handle_userinput(user_question)
        except Exception as e:
            st.write("An error occure: ", e)


    with st.sidebar:
        st.subheader("Upload you PDFs here")
        pdf_docx = st.file_uploader("Upload your PDFs here and click on 'Process", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Loading.."):
                #get the pdf text
                raw_text = get_pdf_text(pdf_docx)
                #st.write(raw_text)
                
                #get the text chunks
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)

                #get the embeddings/vector store
                vectorstore = get_vectorstore(text_chunks)

                #create converstaion chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
            
    

if __name__ == '__main__':
    main()