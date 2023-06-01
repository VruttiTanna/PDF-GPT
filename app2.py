import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import os
import tempfile
from pdf2image import convert_from_bytes
import io
from PIL import Image


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
def process_file(file_bytes):
    if 'OPENAI_API_KEY' not in os.environ:
        st.error('Upload your OpenAI API key')

    loader = PyPDFLoader(file_bytes)
    documents = loader.load()

    embeddings = OpenAIEmbeddings()

    pdfsearch = Chroma.from_documents(documents, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.3),
        retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=True
    )

    return chain


# Streamlit application setup
st.set_page_config(page_title='Chatbot with PDF Support', layout='wide')
st.title('Chatbot with PDF Support')

# Sidebar - OpenAI API Key
st.sidebar.header('OpenAI API Key')
api_key = st.sidebar.text_input('Enter your OpenAI API key')
if api_key:
    set_apikey(api_key)

# Main Section - Chatbot and PDF Upload
col1, col2, col3 = st.beta_columns([2, 1, 1])

# Chatbot Section
with col1:
    st.subheader('Chatbot')
    txt = st.text_input('Enter text and press Enter')
    chat_history = []

    if st.button('Submit'):
        if not txt:
            st.error('Enter text')
        else:
            add_text(chat_history, txt)
            if 'temp_bytes' in st.session_state:
                chain = process_file(st.session_state.temp_bytes)
                context = " ".join([item[0] for item in chat_history])
                prompt_template = "The document mentions {}. What would you like to know about it?"

                if chain.retriever.has_data():
                    result = chain({
                        "question": txt,
                        'chat_history': chat_history,
                        'context': context,
                        'prompt_template': prompt_template
                    }, return_only_outputs=True)

                    chat_history.append((txt, result["answer"]))

                    # Display chat history
                    for idx, (question, answer) in enumerate(chat_history):
                        st.text_area(f'{idx + 1}. User:', question, height=100)
                        st.text_area(f'{idx + 1}. Chatbot:', answer, height=100)
                else:
                    st.error('The uploaded PDF does not contain any searchable content.')

# PDF Upload Section
with col2:
    st.subheader('Upload PDF')
    uploaded_file = st.file_uploader('Upload a PDF', type=".pdf")

    if uploaded_file:
        file_bytes = uploaded_file.read()
        st.session_state.temp_bytes = file_bytes

        st.success('PDF uploaded successfully!')

# PDF Preview Section
with col3:
    st.subheader('PDF Preview')
    if 'temp_bytes' in st.session_state:
        images = convert_from_bytes(st.session_state.temp_bytes, first_page=0, last_page=1)

        if images:
            image = images[0]
            image_stream = io.BytesIO()
            image.save(image_stream, format='PNG')

            st.image(image_stream, caption='PDF Preview', use_column_width=True)
        else:
            st.warning('Unable to preview the PDF.')
    else:
        st.info('Upload a PDF to see the preview.')

