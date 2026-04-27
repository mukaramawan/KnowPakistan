#load pdf
#split into chunks
#create embeddings 
#store in vector database
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

loader = PyPDFLoader(r"C:\Users\mukar\OneDrive\Desktop\RAG Project\1- Docs Loader & Text Splitting\Transformers Article.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = splitter.split_documents(docs)

for i in range(len(chunks)):
    print(f"Chunk {i+1}: {chunks[i].page_content}")
    print("")
    print("")

embedding_model = HuggingFaceEmbeddings(model="hreyulog/embedinggemma_arkts")

vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory="./chroma_db")