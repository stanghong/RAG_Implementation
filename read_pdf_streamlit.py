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
import pprint
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

pdf_folder_path = './docs'
os.listdir(pdf_folder_path)

# Load all PDFs excluding unnecessary files
loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) \
for fn in os.listdir(pdf_folder_path)  if fn not in ['.DS_Store']] 


# Create embedding once and save as vector store
index = VectorstoreIndexCreator().from_loaders(loaders)

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-3.5-turbo")
# %%
st.title("Chat with your PDFs")

# Ensure messages is initialized at the top of your script
st.session_state.setdefault('messages', [{"role": "assistant", "content": "Ask me a question!"}])

def generate_response(input_text):
    # Query the index with sources and get both answer and sources
    response = index.query_with_sources(input_text, llm)
    # Assuming 'response' contains 'answer' and 'sources', adjust based on actual structure
    answer = response.get('answer', "I'm sorry, I couldn't find an answer.")
    sources = response.get('sources', "No source available.")
    return {"answer": answer, "sources": sources}
# %%
# Handle the chat input and response logic
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = generate_response(prompt)
    assistant_response = response.get('answer', "I'm sorry, I couldn't find an answer.")
    source_info = response.get('sources', "No source available.")
    st.session_state.messages.append({"role": "assistant", "content": assistant_response, "source": source_info})
# %%
# Display messages
for message in st.session_state.messages:
    with st.container():
        role = message["role"]
        if role == "assistant":
            st.info(message["content"])
            st.caption("Source: " + message.get("source", ""))
        else:
            st.success(message["content"])

# %%
