import os
import streamlit as st
import pandas as pd

from utils.pdf_loader import load_pdf
from utils.text_splitter import split_documents
from utils.embeddings import get_embedding_model
from utils.vector_store import create_vector_store
from utils.retriever import get_retriever
from utils.llm import get_llm
from utils.chatbot import build_rag_chain

from utils.merchant_data_loader import load_merchant_data
from utils.merchant_chatbot import answer_merchant_question


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------
# Custom CSS
# ---------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a, #111827);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    h1 {
        color: white;
        text-align: center;
        font-size: 46px;
    }
    h2, h3 {
        color: white;
    }
    p, label {
        color: white;
    }
    div[data-testid="stFileUploader"] {
        background: #1E293B;
        border: 2px dashed #00C2A8;
        border-radius: 15px;
        padding: 15px;
    }
    .stTextInput > div > div > input {
        background: #1E293B;
        color: white;
        border-radius: 12px;
        border: 2px solid #00C2A8;
    }
    .stButton > button {
        width: 100%;
        min-height: 45px;
        border: none;
        border-radius: 12px;
        background: #00C2A8;
        color: white;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: #14b8a6;
        color: white;
    }
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------
# Session State
# ---------------------------------------------------

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "merchant_dataframe" not in st.session_state:
    st.session_state.merchant_dataframe = None

if "merchant_filename" not in st.session_state:
    st.session_state.merchant_filename = None


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:
    st.title("🤖 SmartDoc AI")
    st.markdown("---")
    st.write("### Available assistants")
    st.info("📄 PDF question answering")
    st.info("📊 CSV / Excel data questions")
    st.write("### Tech stack")
    st.write(
        """
        - Python and Streamlit
        - LangChain
        - HuggingFace embeddings
        - FAISS vector search
        - Groq LLM
        - Pandas
        """
    )
    st.caption("PDF retrieval and spreadsheet analytics use separate workflows.")


# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.markdown(
    """
    <h1>🤖 SmartDoc AI</h1>
    <p style="text-align:center; font-size:21px; color:#CBD5E1;">
        Ask questions about PDF documents or analyze CSV and Excel data
    </p>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------
# Separate workflows
# ---------------------------------------------------

pdf_tab, spreadsheet_tab = st.tabs(
    ["📄 PDF Q&A", "📊 CSV / Excel Q&A"]
)


# ===================================================
# PDF WORKFLOW — existing RAG implementation
# ===================================================

with pdf_tab:
    left, right = st.columns([1, 2])

    with left:
        st.subheader("📂 Upload PDF")

        uploaded_pdf = st.file_uploader(
            "Choose a PDF document",
            type=["pdf"],
            key="pdf_uploader",
        )

        if uploaded_pdf is not None:
            st.success(f"Selected: {uploaded_pdf.name}")

            if st.button("🚀 Process PDF", key="process_pdf"):
                try:
                    with st.spinner("Loading, splitting, embedding, and indexing PDF..."):
                        os.makedirs("uploads", exist_ok=True)

                        # Use a stable filename to avoid path traversal and
                        # avoid writing arbitrary user-supplied paths.
                        safe_name = os.path.basename(uploaded_pdf.name)
                        pdf_path = os.path.join("uploads", safe_name)

                        with open(pdf_path, "wb") as output_file:
                            output_file.write(uploaded_pdf.getbuffer())

                        documents = load_pdf(pdf_path)
                        if not documents:
                            raise ValueError("No readable pages were found in this PDF.")

                        chunks = split_documents(documents)
                        if not chunks:
                            raise ValueError("No text chunks could be created from this PDF.")

                        embedding_model = get_embedding_model()
                        vector_db = create_vector_store(chunks, embedding_model)

                        st.session_state.vector_db = vector_db
                        st.session_state.pdf_name = uploaded_pdf.name

                    st.success(f"PDF processed successfully: {len(chunks)} chunks indexed.")
                except Exception as exc:
                    st.error(f"Could not process the PDF: {exc}")

    with right:
        st.subheader("💬 Ask questions about your PDF")

        if st.session_state.vector_db is not None:
            st.caption(f"Active PDF: {st.session_state.pdf_name or 'Processed document'}")

            pdf_question = st.text_input(
                "Ask anything from the uploaded PDF...",
                key="pdf_question",
            )

            if st.button("🔍 Get PDF Answer", key="ask_pdf"):
                if not pdf_question.strip():
                    st.warning("Enter a question.")
                else:
                    try:
                        with st.spinner("Retrieving relevant chunks and generating an answer..."):
                            retriever = get_retriever(st.session_state.vector_db)
                            llm = get_llm()
                            rag_chain = build_rag_chain(llm, retriever)
                            response = rag_chain.invoke({"input": pdf_question})

                        st.markdown("### 🤖 AI Answer")
                        st.markdown(response.get("answer", "No answer was returned."))

                        # Show retrieved evidence so the RAG behavior can be inspected.
                        with st.expander("Inspect retrieved PDF context (RAG verification)"):
                            context_docs = response.get("context", [])
                            if context_docs:
                                for index, document in enumerate(context_docs, start=1):
                                    metadata = document.metadata or {}
                                    page = metadata.get("page")
                                    source = metadata.get("source", st.session_state.pdf_name)
                                    page_label = f", page {page + 1}" if isinstance(page, int) else ""
                                    st.markdown(f"**Chunk {index} — {source}{page_label}**")
                                    st.text(document.page_content)
                            else:
                                st.info(
                                    "The chain did not return a context field. "
                                    "Check utils/chatbot.py and the chain output."
                                )
                    except Exception as exc:
                        st.error(f"Could not answer the PDF question: {exc}")
        else:
            st.info("Upload and process a PDF to enable document Q&A.")


# ===================================================
# CSV / EXCEL WORKFLOW
# ===================================================

with spreadsheet_tab:
    left, right = st.columns([1, 2])

    with left:
        st.subheader("📊 Upload tabular data")

        uploaded_table = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            key="merchant_data_uploader",
            help="Supported formats: CSV, XLSX, XLS.",
        )

        if uploaded_table is not None:
            st.caption(f"Selected: {uploaded_table.name}")

            if st.button("🚀 Load spreadsheet", key="load_spreadsheet"):
                try:
                    with st.spinner("Reading and validating spreadsheet..."):
                        dataframe = load_merchant_data(uploaded_table)

                        st.session_state.merchant_dataframe = dataframe
                        st.session_state.merchant_filename = uploaded_table.name

                    st.success(
                        f"Loaded {len(dataframe):,} rows and "
                        f"{len(dataframe.columns):,} columns."
                    )
                except Exception as exc:
                    st.error(f"Could not load the spreadsheet: {exc}")

        if st.session_state.merchant_dataframe is not None:
            st.markdown("---")
            st.markdown("**Dataset details**")
            st.write(f"File: {st.session_state.merchant_filename}")
            st.write(f"Rows: {len(st.session_state.merchant_dataframe):,}")
            st.write(f"Columns: {len(st.session_state.merchant_dataframe.columns):,}")

    with right:
        st.subheader("💬 Ask questions about your data")

        dataframe = st.session_state.merchant_dataframe

        if dataframe is not None:
            st.caption(f"Active dataset: {st.session_state.merchant_filename}")

            # ---------------------------------------------------
            # Merchant Data Dashboard
            # ---------------------------------------------------
            st.markdown("### 📈 Dataset dashboard")

            total_transactions = len(dataframe)
            unique_merchants = (
                dataframe["merchant_id"].nunique(dropna=True)
                if "merchant_id" in dataframe.columns
                else None
            )

            refund_count = None
            if "is_refund" in dataframe.columns:
                refund_values = dataframe["is_refund"]
                refund_count = int(
                    refund_values.astype(str).str.strip().str.lower().isin(
                        ["1", "true", "yes", "y"]
                    ).sum()
                )
            elif "status" in dataframe.columns:
                refund_count = int(
                    dataframe["status"].astype(str).str.strip().str.lower().eq("refund").sum()
                )

            chargeback_count = None
            if "is_chargeback" in dataframe.columns:
                chargeback_values = dataframe["is_chargeback"]
                chargeback_count = int(
                    chargeback_values.astype(str).str.strip().str.lower().isin(
                        ["1", "true", "yes", "y"]
                    ).sum()
                )

            success_count = None
            if "status" in dataframe.columns:
                success_count = int(
                    dataframe["status"].astype(str).str.strip().str.lower().eq("success").sum()
                )

            metric_cols = st.columns(4)
            metric_cols[0].metric("Transactions", f"{total_transactions:,}")
            metric_cols[1].metric(
                "Unique merchants",
                f"{unique_merchants:,}" if unique_merchants is not None else "N/A",
            )
            metric_cols[2].metric(
                "Refunds",
                f"{refund_count:,}" if refund_count is not None else "N/A",
            )
            metric_cols[3].metric(
                "Chargebacks",
                f"{chargeback_count:,}" if chargeback_count is not None else "N/A",
            )

            if success_count is not None:
                success_rate = (success_count / total_transactions * 100) if total_transactions else 0
                st.caption(
                    f"Successful transactions: {success_count:,} "
                    f"({success_rate:.1f}% of all rows)"
                )

            chart_left, chart_right = st.columns(2)

            with chart_left:
                if "status" in dataframe.columns:
                    st.markdown("**Transactions by status**")
                    status_counts = (
                        dataframe["status"].fillna("Missing").astype(str).value_counts()
                    )
                    st.bar_chart(status_counts)

                elif "category" in dataframe.columns:
                    st.markdown("**Transactions by category**")
                    category_counts = (
                        dataframe["category"].fillna("Missing").astype(str).value_counts()
                    )
                    st.bar_chart(category_counts)

            with chart_right:
                if "category" in dataframe.columns:
                    st.markdown("**Transactions by category**")
                    category_counts = (
                        dataframe["category"].fillna("Missing").astype(str).value_counts()
                    )
                    st.bar_chart(category_counts)
                elif "payment_method" in dataframe.columns:
                    st.markdown("**Transactions by payment method**")
                    method_counts = (
                        dataframe["payment_method"].fillna("Missing").astype(str).value_counts()
                    )
                    st.bar_chart(method_counts)

            # Do not add amounts across currencies: that would produce a misleading
            # financial total when a dataset contains multiple currencies.
            if {"amount", "currency"}.issubset(dataframe.columns):
                amount_data = dataframe.copy()
                amount_data["amount"] = pd.to_numeric(amount_data["amount"], errors="coerce")
                currency_summary = (
                    amount_data.dropna(subset=["amount"])
                    .groupby("currency", dropna=False)["amount"]
                    .agg(["count", "sum", "mean"])
                    .reset_index()
                )
                if not currency_summary.empty:
                    st.markdown("**Amount summary by currency**")
                    st.dataframe(currency_summary, use_container_width=True, hide_index=True)

            with st.expander("Preview uploaded data", expanded=True):
                st.dataframe(dataframe.head(20), use_container_width=True)
                st.markdown("**Column names**")
                st.write(list(dataframe.columns))

            table_question = st.text_input(
                "Ask about totals, averages, counts, categories, or records...",
                key="merchant_question",
                placeholder="e.g. What is the average amount paid?",
            )

            if st.button("🔍 Analyze data", key="ask_spreadsheet"):
                if not table_question.strip():
                    st.warning("Enter a question.")
                else:
                    try:
                        with st.spinner("Analyzing data and preparing an explanation..."):
                            llm = get_llm()
                            answer = answer_merchant_question(
                                dataframe,
                                table_question,
                                llm,
                            )

                        st.markdown("### 🤖 Data Answer")
                        if isinstance(answer, str):
                            st.markdown(answer)
                        else:
                            st.write(answer)
                    except Exception as exc:
                        st.error(f"Could not answer the spreadsheet question: {exc}")
        else:
            st.info("Upload and load a CSV or Excel file to enable data Q&A.")
