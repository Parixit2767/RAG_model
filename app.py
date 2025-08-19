import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI  # replace with Gemma2 if you have it
from langchain_ollama import OllamaEmbeddings,OllamaLLM
from langchain.embeddings import HuggingFaceEmbeddings

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="NX/MSC Nastran Debugger", layout="wide")
st.title("🛠️ Nastran Debugging Assistant (RAG)")

# -----------------------------
# Load embeddings & FAISS
# -----------------------------
st.sidebar.header("Settings")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")   # replace with Gemma2 embeddings if available

with st.spinner("Loading FAISS index..."):
    db = FAISS.load_local(
        r"F:\\VS code\\NX-DEBBUGGER-RAG_APP\\faiss_index",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
retriever = db.as_retriever(search_kwargs={"k": 5})

# -----------------------------
# Prompt template
# -----------------------------
prompt_template = """
You are an expert in NX Nastran and MSC Nastran debugging.
Answer the user's question using the retrieved documents. 
If the answer is not in the documents, say "I could not find relevant information".
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template + "\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)

# -----------------------------
# Create RetrievalQA chain
# -----------------------------
llm = OllamaLLM(model="gemma2:2b") # replace with Gemma2 if using
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,         
    retriever=retriever,     
    return_source_documents=True,
    chain_type="stuff",      
    chain_type_kwargs={"prompt": prompt},
    input_key="question" 
    )






# -----------------------------
# User query input
# -----------------------------
query = st.text_input("Enter your Nastran error or question:", "")

if query:
    with st.spinner("Fetching answer..."):
        output = qa_chain({"question": query})
    
    st.subheader("Answer")
    st.write(output)  # main answer






