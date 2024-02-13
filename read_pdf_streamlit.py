# %%

import streamlit as st


import os
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
import pdf2image, pdfminer
# %%
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY  = os.getenv('OPENAI_API_KEY')
# %%
pdf_folder_path = f'./docs'
os.listdir(pdf_folder_path)
# %%
# listcompresion to load all pdfs and exclude files not needed
loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) \
for fn in os.listdir(pdf_folder_path)  if fn not in ['.DS_Store']] 
# %%
# creat embedding once and save as vectorstore
index = VectorstoreIndexCreator().from_loaders(loaders)
# %%
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, 
             temperature=0,
             model_name="gpt-3.5-turbo")
# # res=index.query_with_sources('what is the pdf about', llm)
# res=index.query_with_sources('explain how to do QA pdf with source', llm)
# %%

st.title("ChatGPT-like clone")

def generate_response(input_text):
    response=index.query_with_sources('explain how to do QA pdf with source', llm)
    st.info(response)

with st.form('my_form'):
    text = st.text_area('Enter text:', 'what is the key to do RAG in llm?')
    submitted = st.form_submit_button('Submit')
    # if not openai_api_key.startswith('sk-'):
    #     st.warning('Please enter your OpenAI API key!', icon='⚠')
    # if submitted and openai_api_key.startswith('sk-'):
        # generate_response(text)
    generate_response(text)