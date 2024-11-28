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
from langchain.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory




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
    # Initialize the model with your specific identifier
    llm = HuggingFaceHub(repo_id="gpt-4o-mini-2024-07-18:personal:fine-tune-test:AYMkExVx", model_kwargs={"temperature":0.2, "max_length":512})
  
    # Set up memory for conversation history
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    # Create the conversational retrieval chain using the vector store
    conver_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
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
    #load_dotenv()

    st.set_page_config(page_title='Chat with PDFs')
    st.write(css, unsafe_allow_html=True)
    st.title("Oregon Gov. Assistance Bot")

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


            
    

if __name__ == '__main__':
    main()