# %%
# Import necessary libraries
import os
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
# %%
# Load environment variables
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Define the PDF folder path
pdf_folder_path = './docs'

# Ensure the PDF folder path exists
if not os.path.exists(pdf_folder_path):
    raise FileNotFoundError(f"The directory {pdf_folder_path} does not exist.")

# Load all PDF files from the directory, excluding any that are not PDFs
loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn))
           for fn in os.listdir(pdf_folder_path) if fn.endswith('.pdf')]
# %%
# Function to load documents from a given loader
def load_documents(loader):
    return loader.load()

# Function to chunk documents
def chunk_documents(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(data)

# # Example usage for a single file - adjust as needed for your use case
# if loaders:  # Check if there are any loaders
#     data = load_documents(loaders[0])  # Load document from the first loader
#     texts = chunk_documents(data)  # Chunk documents
#     print(f'Now you have {len(texts)} chunks of text from the document.')

#     # Initialize embedding and vector storage
#     embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)  # Ensure to pass the API key if needed
#     persist_directory = 'db'
#     vectordb = Chroma.from_documents(documents=texts,
#                                      embedding=embedding,
#                                      persist_directory=persist_directory)
# else:
#     print("No PDF files found in the specified directory.")
# Example usage for a single file - adjust as needed for your use case
if loaders:  # Check if there are any loaders
    data = load_documents(loaders[0])  # Load document from the first loader
    texts = chunk_documents(data)  # Chunk documents
    print(f'Now you have {len(texts)} chunks of text from the document.')

    # Initialize embedding
    embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)  # Ensure to pass the API key if needed
    persist_directory = 'db'

    # Check if ChromaDB exists before creating
    db_path = os.path.join(persist_directory, 'f7614288-bfc0-43fc-b404-d53419d359ff')  # Assuming 'chromadb' is the name of the database
    if os.path.exists(db_path):
        print("Loading existing ChromaDB vector store...")
        vectordb = Chroma(persist_directory=persist_directory ,
embedding_function=embedding)  # Adjust method based on actual API
    else:
        print("Creating new ChromaDB vector store...")
        vectordb = Chroma.from_documents(documents=texts,
                                         embedding=embedding,
                                         persist_directory=persist_directory)
else:
    print("No PDF files found in the specified directory.")
# %%
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
# %%
llm = ChatOpenAI(temperature=0.0)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "document_separator": "<<<<>>>>>"
    }
    ,return_source_documents=True,
)
query = "what is RAG"
result = qa({"query": query})
#
# %%
print(f'Query: {result["query"]}\n')
print(f'Result: {result["result"]}\n')
print(f'Context Documents: ')
for srcdoc in result["source_documents"]:
      print(f'{srcdoc}\n')
# %%
