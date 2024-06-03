import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Set up environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# Load and embed the documents
def embed_and_save_documents():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFDirectoryLoader("./LEGAL-DATA")
    print("Loader initialised")
    docs = loader.load()
    print("Loading the docs")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    print("Splitting the docs")
    
    # Ensure metadata includes the source file name
    for doc in final_documents:
        if 'source' in doc.metadata:
            source_file = doc.metadata['source']
            doc.metadata['source'] = os.path.basename(source_file)
        else:
            # If source metadata is not present, add it
            doc.metadata['source'] = os.path.basename(loader.directory)
    
    # Ensure the payload size is within limits by batching the documents
    batch_size = 100  # Adjust batch size as needed
    batched_documents = [final_documents[i:i + batch_size] for i in range(0, len(final_documents), batch_size)]
    vector_stores = []
    for batch in batched_documents:
        vector_store = FAISS.from_documents(batch, embeddings)
        vector_stores.append(vector_store)
    print("created batched documents")
    
    # Merge the vector stores
    vectors = vector_stores[0]
    for vector_store in vector_stores[1:]:
        vectors.merge_from(vector_store)
    print("merged the vectors")
    
    # Save the vector store to disk
    vectors.save_local("my_vector_store")
    print("vectors saved")

embed_and_save_documents()
