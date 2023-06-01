import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import os
import tempfile


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
def process_file(file_path):
    if 'OPENAI_API_KEY' not in os.environ:
        st.error('Upload your OpenAI API key')

    loader = PyPDFLoader(file_path)
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
st.title('Chatbot with PDF Support')

# OpenAI API Key
st.sidebar.header('OpenAI API Key')
api_key = st.sidebar.text_input('Upload your OpenAI API key')
if api_key:
    set_apikey(api_key)

# Chatbot and Image Display
st.subheader('Chatbot')
txt = st.text_input('Enter text and press enter')
chat_history = []

st.subheader('Upload PDF')
uploaded_file = st.file_uploader('Upload a PDF', type=".pdf")
show_img = st.empty()

submit_btn = st.button('Submit')

if submit_btn:
    if not uploaded_file:
        st.error('Upload a PDF file')
    else:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(uploaded_file.read())

        add_text(chat_history, txt)
        chain = process_file(temp_path)

        # Set context and prompt template
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
                st.text(f"{idx + 1}. User: {question}")
                st.text(f"   Chatbot: {answer}")

            show_img.image(result["image"], caption="Document Page", use_column_width=True)
        else:
            st.error('The uploaded PDF does not contain any searchable content.')

        os.remove(temp_path)
