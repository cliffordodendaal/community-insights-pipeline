# src/app.py
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Community Insights — Census RAG", layout="centered")

@st.cache_resource
def load_rag(index_path: str = "faiss_index", k: int = 5):
    """
    Load the FAISS index once and return a callable that accepts a question
    and returns the LLM response string.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    def ask(question: str):
        if not question or not question.strip():
            return "Please enter a question."
        result = rag_chain.invoke(question)
        # result may be a ChatMessage or similar; handle content attribute if present
        return getattr(result, "content", str(result))

    return ask

st.title("Community Insights — Census RAG")
st.write("Ask questions about the embedded census data. Make sure you have already run the ingestion step so `faiss_index/` exists.")

# sample prompts
with st.expander("Sample prompts"):
    st.write("- Which municipalities in KwaZulu-Natal have the lowest access to piped water?")
    st.write("- Show the top three municipalities by population without piped water access.")
    st.write("- List key statistics for eThekwini municipality from the dataset.")

question = st.text_input("Ask a question about the census", "")
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            try:
                with st.spinner("Loading models and index (cached after first load)..."):
                    ask = load_rag()  # cached resource
                    answer = ask(question)
                st.markdown("**Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
with col2:
    if st.button("Use sample question"):
        question = "Which municipalities in KwaZulu-Natal have the lowest access to piped water?"
        st.experimental_rerun()

st.caption("Tip: run `py src/ingest.py` first if you haven't built the faiss_index yet.")
