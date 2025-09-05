from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from config.settings import settings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=settings.PERSIST_DIR
)
vectorstore.persist()
retriever = vectorstore.as_retriever()