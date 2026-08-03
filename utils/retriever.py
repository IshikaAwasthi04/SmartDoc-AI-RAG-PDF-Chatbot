def get_retriever(vector_store, k=3):
    """
    Create a retriever from the FAISS vector store.

    Args:
        vector_store: FAISS vector database
        k (int): Number of relevant chunks to retrieve

    Returns:
        Retriever object
    """

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    return retriever