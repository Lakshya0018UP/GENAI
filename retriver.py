from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_retriver():
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_db=Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    return vector_db.as_retriever(search_kwargs={"k":3})