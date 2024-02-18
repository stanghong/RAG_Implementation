# %%
import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# %%
# PDF processing
pdf_folder_path = './data/PDFs'
if not os.path.exists(pdf_folder_path):
    raise FileNotFoundError(f"The directory {pdf_folder_path} does not exist.")

def load_and_chunk_pdfs(pdf_folder_path):
    loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn))
               for fn in os.listdir(pdf_folder_path) if fn.endswith('.pdf')]
    all_texts = []
    for loader in loaders:
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(data)
        all_texts.extend(texts)
    return all_texts

embeddings = OllamaEmbeddings(model="mistral")
vectorstore_directory = 'vdb_multipdfs'
if not os.path.exists(vectorstore_directory) or not os.listdir(vectorstore_directory):
    all_texts = load_and_chunk_pdfs(pdf_folder_path)
    # Embedding and storing
    vectorstore = Chroma.from_documents(documents=all_texts,
                                        embedding=embeddings,
                                        persist_directory=vectorstore_directory)
else:
    vectorstore = Chroma(persist_directory=vectorstore_directory,
                         embedding_function=embeddings)
# %%
# LLM and QA Chain setup
from langchain.schema import LLMResult
from langchain.callbacks.base import BaseCallbackHandler

class GenerationStatisticsCallback(BaseCallbackHandler):
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(response.generations[0][0].generation_info)
        
callback_manager = CallbackManager([StreamingStdOutCallbackHandler(), GenerationStatisticsCallback()])

llm = Ollama(base_url="http://localhost:11434",
             model="mistral",
             verbose=True,
             callback_manager=callback_manager)
# %%
def generate_response(question):
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, please think rationally and answer from your own knowledge base 

{context}

Question: {question}
"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs=chain_type_kwargs 
    ,return_source_documents=True,
)
    res = qa({"query": question})
    return res

# %%
# Streamlit interaction
st.title("Chat with your PDFs")
st.session_state.setdefault('messages', [{"role": "assistant", "content": "Ask me a question!"}])

# Handle the chat input and response logic
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = generate_response(prompt)
    assistant_response = response.get('result', "I'm sorry, I couldn't find an result.")
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
   
    if 'source_documents' in response and response['source_documents']:
    # Initialize an empty list to hold the source metadata
        sources_metadata = []

        # Loop through each document in the source documents
        for doc in response['source_documents']:
            # Assuming doc is an instance of Document and has a method or property to access metadata
            # And assuming metadata is an object or dictionary that includes a 'source' key or attribute
            try:
                # Attempt to access the source information from the metadata
                source_info = doc.metadata['source']  # Use this if metadata is a dictionary
                # If metadata is accessed through methods or properties, adjust the above line accordingly
                sources_metadata.append(source_info)
            except AttributeError:
                # Handle cases where doc does not have a metadata attribute or metadata does not have a 'source' key
                print(f"Document {doc} does not have the expected metadata structure.")

        # Concatenate all source metadata into one string, separated by new lines
        sources_concatenated = '\n'.join(sources_metadata)
    
    # Append the concatenated source metadata to the assistant's message
    st.session_state.messages.append({"role": "assistant", "content": f"Sources:\n{sources_concatenated}"})


# Display messages
for message in st.session_state.messages:
    with st.container():
        role = message["role"]
        # if role == "assistant":
        #     st.info(message["content"])
        # else:
        #     st.success(message["content"])
        if role == "assistant":
            st.info(message["content"])
            # st.info(message["source"])
        else:
            st.success(message["content"])
