"""
modules.py
Reusable building blocks for the RAG pipeline.
Includes: load_pdf, chunk_documents, embed_chunks, query_chunks
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

def load_pdf(path):
    """Load a PDF and return LangChain documents"""
    loader = PyPDFLoader(path)
    return loader.load()

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks for embedding"""
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def embed_chunks(chunks, index_path="faiss_index"):
    """Embed chunks and store them in a FAISS vectorstore"""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

def query_chunks(question, index_path="faiss_index"):
    """Query the FAISS vectorstore and return an answer"""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain.invoke(question).content

