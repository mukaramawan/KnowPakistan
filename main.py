from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
load_dotenv()

embedding_model = HuggingFaceEmbeddings(model="hreyulog/embedinggemma_arkts")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 100, "lambda_mult": 0.5}, 
)
#  lambda mult controls the balance between relevance and diversity

llm = ChatGroq(model="groq/compound-mini", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that provides information based on the provided context. If the context does not contain the answer, say I don't know."""),
    ("human", """Context:\n\n{context}\n\n Question: {question}""")])

print("Press 0 to exit.")
while True:
    question = input("Enter your question: ")
    if question == "0":
        break
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    response = llm.invoke(prompt.format(context=context, question=question))
    print("\nAI:", response.content) 

