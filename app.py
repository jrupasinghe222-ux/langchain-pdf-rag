import os
import tempfile

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_core.prompts import ChatPromptTemplate


st.set_page_config(
    page_title="LangChain PDF RAG",
    page_icon="📄",
)

st.title("📄 LangChain PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions about its contents."
)


# Store the retrieval chain between Streamlit reruns.
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "processed_file_name" not in st.session_state:
    st.session_state.processed_file_name = None


uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)


if uploaded_file is not None:

    process_button = st.button(
        "Process PDF",
        type="primary",
    )

    if process_button:

        temporary_pdf_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as temporary_file:

                temporary_file.write(
                    uploaded_file.getvalue()
                )

                temporary_pdf_path = temporary_file.name

            with st.spinner("Reading PDF..."):

                loader = PyPDFLoader(
                    temporary_pdf_path
                )

                documents = loader.load()

            if not documents:
                raise ValueError(
                    "No readable text was found in the PDF."
                )

            with st.spinner("Splitting document into chunks..."):

                text_splitter = (
                    RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=100,
                    )
                )

                chunks = text_splitter.split_documents(
                    documents
                )

            if not chunks:
                raise ValueError(
                    "No text chunks could be created."
                )

            with st.spinner("Creating embeddings..."):

                embedding_model = HuggingFaceEmbeddings(
                    model_name=(
                        "sentence-transformers/"
                        "all-MiniLM-L6-v2"
                    )
                )

                vector_store = FAISS.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                )

            # Retrieve the three most relevant chunks.
            retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": 3,
                }
            )

            llm = ChatOllama(
                model="llama3",
                temperature=0,
            )

            system_prompt = """
You are a question-answering assistant.

Answer the user's question using only the provided context.

If the answer cannot be found in the context, say:
"I couldn't find that information in the uploaded PDF."

Do not invent information.

Context:
{context}
"""

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        system_prompt,
                    ),
                    (
                        "human",
                        "{input}",
                    ),
                ]
            )

            # This chain joins the retrieved documents and
            # inserts them into the {context} prompt variable.
            document_chain = create_stuff_documents_chain(
                llm=llm,
                prompt=prompt,
            )

            # This chain:
            # 1. receives the question,
            # 2. retrieves relevant chunks,
            # 3. sends those chunks to the document chain.
            rag_chain = create_retrieval_chain(
                retriever=retriever,
                combine_docs_chain=document_chain,
            )

            st.session_state.rag_chain = rag_chain
            st.session_state.processed_file_name = (
                uploaded_file.name
            )

            st.success(
                f"PDF processed successfully. "
                f"Created {len(chunks)} chunks."
            )

        except Exception as error:
            st.error(
                f"Could not process the PDF: {error}"
            )

        finally:
            if (
                temporary_pdf_path
                and os.path.exists(temporary_pdf_path)
            ):
                os.remove(temporary_pdf_path)


if st.session_state.rag_chain is not None:

    # st.info(
    #     "Current PDF: "
    #     f"{st.session_state.processed_file_name}"
    # )

    question = st.text_input(
        "Ask a question about the PDF"
    )

    ask_button = st.button("Ask")

    if ask_button:

        if not question.strip():
            st.warning(
                "Enter a question first."
            )

        else:
            try:
                with st.spinner(
                    "Searching the PDF and generating an answer..."
                ):

                    response = (
                        st.session_state.rag_chain.invoke(
                            {
                                "input": question.strip(),
                            }
                        )
                    )

                st.subheader("Answer")

                st.write(
                    response["answer"]
                )

                retrieved_documents = response.get(
                    "context",
                    [],
                )

                if retrieved_documents:

                    with st.expander(
                        "View retrieved PDF sections"
                    ):

                        for number, document in enumerate(
                            retrieved_documents,
                            start=1,
                        ):

                            page_number = (
                                document.metadata.get(
                                    "page",
                                    0,
                                )
                                + 1
                            )

                            st.markdown(
                                f"**Source {number} — "
                                f"Page {page_number}**"
                            )

                            st.write(
                                document.page_content
                            )

                            if number < len(
                                retrieved_documents
                            ):
                                st.divider()

            except Exception as error:
                st.error(
                    "Could not generate an answer. "
                    f"Error: {error}"
                )


elif uploaded_file is None:

    st.info(
        "Upload and process a PDF to begin."
    )