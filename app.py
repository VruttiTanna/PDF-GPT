import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
import os

import fitz
from PIL import Image

# Global variables
COUNT, N = 0, 0
chat_history = []
chain = None

# Function to set the OpenAI API key
def set_apikey(api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    st.write('OpenAI API key is set')

# Function to enable the API key input box
def enable_api_box():
    api_key = st.text_input('Enter OpenAI API key')
    if api_key:
        set_apikey(api_key)

# Function to add text to the chat history
def add_text(text):
    if not text:
        st.error('Enter text')
        return
    global chat_history
    chat_history.append((text, ''))
    st.write(chat_history)

# Function to process the PDF file and create a conversation chain
def process_file(file):
    if 'OPENAI_API_KEY' not in os.environ:
        st.error('Upload your OpenAI API key')
        return None

    loader = PyPDFLoader(file.name)
    documents = loader.load()

    embeddings = OpenAIEmbeddings()
    pdfsearch = Chroma.from_documents(documents, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.3),
        retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=True
    )
    return chain

# Function to generate a response based on the chat history and query
def generate_response(query, file):
    global COUNT, N, chat_history, chain

    if not file:
        st.error('Upload a PDF')
        return

    if COUNT == 0:
        chain = process_file(file)
        COUNT += 1

    result = chain({"question": query, 'chat_history': chat_history}, return_only_outputs=True)
    chat_history.append((query, result["answer"]))
    N = list(result['source_documents'][0])[1][1]['page']

    st.write(chat_history)

# Function to render a specific page of a PDF file as an image
def render_file(file):
    global N
    doc = fitz.open(file.name)
    page = doc[N]
    # Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    st.image(image)

# Streamlit application setup
st.title('Chatbot with PDF Support')

# API Key Input
st.sidebar.header('OpenAI API Key')
enable_api_box()

# Chatbot and Image Display
st.subheader('Chatbot')
chat_input = st.text_input('Enter text and press enter')
chat_history_output = st.empty()

st.subheader('Upload PDF')
uploaded_file = st.file_uploader('Upload a PDF', type=".pdf")
image_output = st.empty()

# Perform actions on text input and PDF upload
if st.button('Submit'):
    add_text(chat_input)

if uploaded_file is not None:
    generate_response(chat_input, uploaded_file)
    render_file(uploaded_file)
