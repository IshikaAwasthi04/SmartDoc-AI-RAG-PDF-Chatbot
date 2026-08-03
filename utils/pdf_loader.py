from langchain_community.document_loaders import PyPDFLoader
import os

def load_pdf(file_path):
    """
    Load a PDF and return its pages as LangChain Document objects.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    return documents