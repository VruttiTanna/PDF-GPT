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
from io import BytesIO

# Global variables
COUNT, N = 0, 0
chat_history = []
chain = None

# Function to set the OpenAI API key
def set_apikey(api_key):
    os.environ['OPENAI_API_KEY'] = api_key

# Function to add text to the chat history
def add_text(history, text):
    if not text:
        st.error('Enter text')
    history.append((text, ''))
    return history

# Function to process the PDF file and create a conversation chain
def process_file(file):
    if 'OPENAI_API_KEY' not in os.environ:
        st.error('Upload your OpenAI API key')

    loader = PyPDFLoader(file.name)
    documents = loader.load()

    embeddings = OpenAIEmbeddings()
    pdfsearch = Chroma.from_documents(documents, embeddings)

    chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.3),
                                                  retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
                                                  return_source_documents=True)
    return chain

# Function to generate a response based on the chat history and query
def generate_response(history, query, btn):
    global COUNT, N, chat_history, chain

    if not btn:
        st.error('Upload a PDF')
    if COUNT == 0:
        chain = process_file(btn)
        COUNT += 1

    result = chain({"question": query, 'chat_history': chat_history}, return_only_outputs=True)
    chat_history.append((query, result["answer"]))
    N = list(result['source_documents'][0])[1][1]['page']

    for char in result['answer']:
        history[-1] = (history[-1][0], history[-1][1] + char)
        yield history, ''

# Function to render a specific page of a PDF file as an image
def render_file(btn):
    global N
    try:
        pdf_bytes = BytesIO(btn.read())
        with fitz.open("pdf", pdf_bytes) as doc:
            page = doc[N]
            # Render the page as a PNG image with a resolution of 300 DPI
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image
    except FileNotFoundError:
        raise st.Error('PDF file not found. Please make sure the file exists and check the file path.')

# Streamlit application setup
st.title('Chatbot with PDF Support')

# OpenAI API Key
st.sidebar.header('OpenAI API Key')
api_key = st.sidebar.text_input('Upload your OpenAI API key')
if api_key:
    set_apikey(api_key)

# Chatbot and Image Display
st.subheader('Chatbot')
chat_history_output = st.empty()
txt = st.text_input('Enter text and press enter')
chat_history = []

st.subheader('Upload PDF')
btn = st.file_uploader('Upload a PDF', type=".pdf")
show_img = st.empty()

submit_btn = st.button('Submit')

if submit_btn:
    add_text(chat_history, txt)
    for _, _ in generate_response(chat_history, txt, btn):
        pass
    image = render_file(btn)
    if image:
        show_img.image(image)
