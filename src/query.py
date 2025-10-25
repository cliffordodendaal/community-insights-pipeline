"""
query.py
Load the FAISS vectorstore and answer a natural-language question using the embedded census data.
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

def query_vectorstore(question):
    print("Loading vectorstore...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    print("Searching for relevant chunks...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)

    print("Running RAG chain...")
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    answer = rag_chain.invoke(question)

    print("\n--- Question ---")
    print(question)
    print("\n--- Answer ---")
    print(answer.content)

if __name__ == "__main__":
    query = input("Enter your question: ")
    query_vectorstore(query)
