import gradio as gr
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
chain = ''
enable_box = gr.Textbox.update(value=None, placeholder='Upload your OpenAI API key', interactive=True)
disable_box = gr.Textbox.update(value='OpenAI API key is Set', interactive=False)

# Function to set the OpenAI API key
def set_apikey(api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    return disable_box

# Function to enable the API key input box
def enable_api_box():
    return enable_box

# Function to add text to the chat history
def add_text(history, text):
    if not text:
        raise gr.Error('Enter text')
    history = history + [(text, '')]
    return history

# Function to process the PDF file and create a conversation chain
def process_file(file):
    if 'OPENAI_API_KEY' not in os.environ:
        raise gr.Error('Upload your OpenAI API key')

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
        raise gr.Error(message='Upload a PDF')
    if COUNT == 0:
        chain = process_file(btn)
        COUNT += 1
    
    result = chain({"question": query, 'chat_history': chat_history}, return_only_outputs=True)
    chat_history += [(query, result["answer"])]
    N = list(result['source_documents'][0])[1][1]['page']

    for char in result['answer']:
        history[-1][-1] += char
        yield history, ''

# Function to render a specific page of a PDF file as an image
def render_file(file):
    global N
    doc = fitz.open(file.name)
    page = doc[N]
    # Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image

# Gradio application setup
demo = gr.Interface(
    inputs=[gr.Textbox(placeholder='Enter text and press enter')],
    outputs=[gr.Textbox(label='Chatbot', disabled=True), gr.Textbox(label='Query', disabled=True)],
    title='PDF Chatbot',
    layout='vertical',
    theme='default'
)

# Set the OpenAI API key and handle interactions
def set_api_key(api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    return "OpenAI API key is set."

demo.add_textbox('Enter OpenAI API key:', type='password', command=set_api_key)

# Perform actions on text input and PDF upload
def process_text(text):
    chat_history.append(text)
    query = chat_history[-1]
    result = chain({"question": query, 'chat_history': chat_history}, return_only_outputs=True)
    chat_history.append(result['answer'])
    return chat_history[-2:]

demo.add_textbox('Enter text:', command=process_text)

# PDF upload
def process_pdf(file):
    global chain, COUNT
    if COUNT == 0:
        chain = process_file(file)
        COUNT += 1
    return 'PDF uploaded.'

demo.add_file_handler('Upload PDF:', process_pdf)

# Launch the Gradio interface
demo.launch()
