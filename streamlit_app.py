import time
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA

import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

groq_api_key= st.secrets["groq_api_key"]["my_key"]

if not os.path.exists('pdfFiles'):
    os.mkdir("pdfFiles")
    
if not os.path.exists('vectorDb'):
    os.mkdir("vectorDb")
    
    
if 'template' not in st.session_state:
    st.session_state.template = """
    You are a knowledgeable chatbot, here to help with the questions of the user. Your tone should be professional and informative.

   Context: {context}
   History: {history}


   User: {question}
   Chatbot:"""
    
if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history","context","question"],
        template= st.session_state.template
    )
    
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key ="history",
        return_key =True,
        input_key ="question"
    )

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(persist_directory = "vectorDb",
                                          embedding_function = OpenAIEmbeddings())
    
if 'llm' not in st.session_state:
    st.session_state.llm = ChatGroq(groq_api_key = groq_api_key,
               model_name = "Llama3-8b-8192")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    


st.title("Query your PDFs")
uploaded_files = st.file_uploader("Choose PDF file(s)",type = "pdf", accept_multiple_files=True)
#st.text(uploaded_files)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

if (uploaded_files != [] and uploaded_files is not None):
    #st.text("Files Uploaded Successfully")
    for uploaded_file in uploaded_files:
        #st.text("Name of file to be uploaded : " + uploaded_file.name)
        #st.text(os.path.exists(("pdfFiles/"+uploaded_file.name)))
        if not os.path.exists("pdfFiles/" + uploaded_file.name):
            with st.status("Saving your file..."):
                byte_file = uploaded_file.read()
                file = open("pdfFiles/" + uploaded_file.name, 'wb')
                file.write(byte_file)
                file.close()
                
                loader = PyPDFLoader("pdfFiles/"+ uploaded_file.name)
                data = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1500,
                chunk_overlap = 200,
                length_function = len 
                )
                
                all_splits = text_splitter.split_documents(data)
                
                st.session_state.vectorstore = Chroma.from_documents(
                    documents= all_splits,
                    embedding= OpenAIEmbeddings()
                )
                
                st.session_state.vectorstore.persist()
            
    st.session_state.retriever = st.session_state.vectorstore.as_retriever()
    
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm = st.session_state.llm,
            chain_type = "stuff",
            retriever = st.session_state.retriever,
            verbose = True,
            chain_type_kwargs = {
                "verbose" : True,
                "prompt"  : st.session_state.prompt, 
                "memory"  : st.session_state.memory
            }
        )
        
    if user_input := st.chat_input("You:", key="user_input"):
       user_message = {"role": "user", "message": user_input}
       st.session_state.chat_history.append(user_message)
       with st.chat_message("user"):
           st.markdown(user_input)


       with st.chat_message("assistant"):
           with st.spinner("Assistant is typing..."):
               response = st.session_state.qa_chain(user_input)
           message_placeholder = st.empty()
           full_response = ""
           for chunk in response['result'].split():
               full_response += chunk + " "
               time.sleep(0.05)
               # Add a blinking cursor to simulate typing
               message_placeholder.markdown(full_response + "▌")
           message_placeholder.markdown(full_response)


       chatbot_message = {"role": "assistant", "message": response['result']}
       st.session_state.chat_history.append(chatbot_message)


else:
   st.write("Please upload a PDF file to start the chatbot")
            
