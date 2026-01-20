from langchain_community.document_loaders import PyMuPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
def ingestion_service():
    dir_load=DirectoryLoader(
            "./data",
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=False
        )
    pdf_documents=dir_load.load()
    # return pdf_documents


    #Chuking
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )
    document_chunks=text_splitter.split_documents(pdf_documents)


    # Embeddings
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    #Send the embedding to local folder

    vector_db=Chroma.from_documents(
        documents=document_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print(f"Successfully ingested {len(pdf_documents)} documents into vector store.")


if __name__=="__main__":
    ingestion_service()