import os
import streamlit as st

from utils.pdf_loader import load_pdf
from utils.text_splitter import split_documents
from utils.embeddings import get_embedding_model
from utils.vector_store import create_vector_store
from utils.retriever import get_retriever
from utils.llm import get_llm
from utils.chatbot import build_rag_chain

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------
# Custom CSS
# ---------------------------------------------------

st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0f172a,#111827);
}

/* Hide Streamlit Menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

h1{
    color:white;
    text-align:center;
    font-size:50px;
}

h3{
    color:white;
}

p,label{
    color:white;
}

/* Upload Box */
div[data-testid="stFileUploader"]{
    background:#1E293B;
    border:2px dashed #00C2A8;
    border-radius:15px;
    padding:15px;
}

/* Text Box */
.stTextInput>div>div>input{
    background:#1E293B;
    color:white;
    border-radius:12px;
    border:2px solid #00C2A8;
}

/* Buttons */
.stButton>button{
    width:100%;
    height:50px;
    border:none;
    border-radius:12px;
    background:#00C2A8;
    color:white;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#14b8a6;
}

/* Success Message */
div[data-testid="stAlert"]{
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:

    st.title("🤖 SmartDoc AI")

    st.markdown("---")

    st.success("✅ Upload PDF")

    st.info("💬 Ask Questions")

    st.info("🧠 Powered by Llama")

    st.info("📚 FAISS Vector Database")

    st.markdown("---")

    st.write("### Tech Stack")

    st.write("""
- Python
- Streamlit
- LangChain
- HuggingFace
- FAISS
- Groq LLM
""")

# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.markdown(
"""
<h1>🤖 SmartDoc AI</h1>

<p style='text-align:center;
font-size:22px;
color:#CBD5E1;'>

Chat with your PDF Documents using AI

</p>
""",
unsafe_allow_html=True
)

# ---------------------------------------------------
# Session State
# ---------------------------------------------------

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# ---------------------------------------------------
# Two Column Layout
# ---------------------------------------------------

left, right = st.columns([1,2])

# ===================================================
# LEFT COLUMN
# ===================================================

with left:

    st.subheader("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        os.makedirs("uploads", exist_ok=True)

        pdf_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("PDF Uploaded Successfully!")

        if st.button("🚀 Process PDF"):

            with st.spinner("Processing..."):

                documents = load_pdf(pdf_path)

                chunks = split_documents(documents)

                embedding_model = get_embedding_model()

                vector_db = create_vector_store(
                    chunks,
                    embedding_model
                )

                st.session_state.vector_db = vector_db

            st.success("PDF Ready!")

# ===================================================
# RIGHT COLUMN
# ===================================================

with right:

    st.subheader("💬 Ask Questions")

    if st.session_state.vector_db is not None:

        question = st.text_input(
            "Ask anything from the uploaded PDF..."
        )

        if st.button("🔍 Get Answer"):

            if question.strip() == "":
                st.warning("Enter a question.")

            else:

                with st.spinner("Thinking..."):

                    retriever = get_retriever(
                        st.session_state.vector_db
                    )

                    llm = get_llm()

                    rag_chain = build_rag_chain(
                        llm,
                        retriever
                    )

                    response = rag_chain.invoke(
                        {
                            "input": question
                        }
                    )

                st.markdown(
                    f"""
<div style="
background:#1E293B;
padding:25px;
border-radius:15px;
border-left:6px solid #00C2A8;
color:white;
font-size:18px;
">

<h3>🤖 AI Answer</h3>

{response["answer"]}

</div>
""",
                    unsafe_allow_html=True
                )

    else:

        st.info("👈 Upload and Process a PDF first.")