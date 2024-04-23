import os
from pathlib import Path
import streamlit as st
import nest_asyncio

import llama_index.embeddings.cohere
from langchain_community.embeddings import CohereEmbeddings
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.legacy.llms import Cohere
from llama_index.legacy.vector_stores import PGVectorStore
from typing import Optional

nest_asyncio.apply()

from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from llama_index.legacy import SimpleDirectoryReader

os.environ["COHERE_API_KEY"] = "2DNlMKIjntYyI9fflsvWJ9Nqn0cfZSyZUV92J2o6"

def connection():
    # See docker command above to launch a postgres instance with pgvector enabled.
    connection = "postgresql+psycopg://nouha:nouha@localhost:5432/rag_db"  # Uses psycopg3!
    collection_name = "my_docs"
    embeddings = CohereEmbeddings()

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
        )
    return vectorstore

def read_file_to_doc(fileNames: [str]):
    documents = SimpleDirectoryReader(input_files=fileNames).load_data()
    # Combine all document texts into a single string separated by new lines
    combined_text = "\n".join([doc.text for doc in documents])

    # Creating a Document object with the combined text
    # Ensure to match this parameter name with what Document class constructor expects
    doc = Document(page_content=combined_text)
    return doc

def create_storage_context(host: str = "localhost", port: str = "5432", username: str = "nouha",
                           password: str = "nouha", db_name: str = None, table: str = None, embed_dim: int = 1024):
    pgVectorStore = PGVectorStore.from_params(
        host=host,
        port=port,
        user=username,
        password=password,
        database=db_name,
        table_name=table,
        embed_dim=embed_dim
    )
    return StorageContext.from_defaults(vector_store=pgVectorStore)

def to_vector_store_index(documents: [Document], storageContext: Optional[str] = None):
    return VectorStoreIndex.from_documents(documents=documents, storage_context=storageContext)


def initialize_session_storage():
    if "history" not in st.session_state:
        st.session_state.history = []


def read_from_session_state():
    for hist in st.session_state.history:
        with st.chat_message(hist["role"]):
            st.write(hist["query"])

def define_global_settings(setting: str, value: str):
    if setting == "llm":
        Settings.llm = value
    elif setting == "embed_model":
        Settings.embed_model = value


@st.cache_resource(show_spinner=False)
def query_engine_from_doc(documents: [str]):
    # load_documents
    doc = read_file_to_doc(documents)
    # create embedding
    embed_mod = CohereEmbeddings(model = "embed-multilingual-v2.0")
    # create our llm mistral from ollama
    llm = Cohere(model="command")
    # set Settings
    define_global_settings("llm", llm)
    define_global_settings("embed_model", embed_mod)
    # create storage context
    pgvectore_store_context = create_storage_context(db_name="rag_db", table="test_table", embed_dim=768)
    # document to vector store
    vect_stor = to_vector_store_index([doc], pgvectore_store_context)
    # query engine
    return vect_stor.as_query_engine()


def save_msg(role: str, message: str):
    st.session_state.history.append({"role": role, "query": message})


def save_file(where: str = "./files", file=None):
    path = Path(where, file.name)
    with open(path, mode="wb") as f:
        f.write(file.getvalue())
    return path.exists()




