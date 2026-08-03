from langchain_community.vectorstores import FAISS
import os


def create_vector_store(chunks, embedding_model):

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    return vector_db


def save_vector_store(vector_db):

    os.makedirs("vectorstore", exist_ok=True)

    vector_db.save_local("vectorstore")


def load_vector_store(embedding_model):

    vector_db = FAISS.load_local(
        "vectorstore",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_db