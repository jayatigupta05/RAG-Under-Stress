from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import os
import logging


logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Check if FAISS index already exists
if os.path.exists("faiss_index"):
    print("Loading existing FAISS index...")
    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("Building FAISS index (first run only)...")

    # 1. Load documents
    docs = []
    for file in os.listdir("data"):
        if file.endswith(".txt"):
            loader = TextLoader(f"data/{file}")
            docs.extend(loader.load())

    # 2. Chunk (smaller + faster)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # 3. Create and save vector DB
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")

# 4. LLM
llm = Ollama(model="phi3")

print("Warming up model...")
llm.invoke("Hello")

# 5. Query loop
while True:
    query = input("\nAsk something: ")

    results = db.similarity_search_with_score(query, k=3)

    filtered_results = []

    if len(filtered_results) == 0:
        context = ""
    else:
        context = "\n\n".join([doc.page_content for doc in filtered_results])

    # Optional: show retrieved chunks (great for demo)
    # for i, r in enumerate(results):
    #     print(f"\n--- Chunk {i+1} ---\n{r.page_content}")

    prompt = f"""
    Answer ONLY using the context below.
If the answer is not there, say "I don't know.
- Use ONLY the provided context
- If the context is empty OR does not contain the answer, respond EXACTLY with: I don't know
- Do NOT use any other knowledge outside of what is provided in the context

Context:
{context}

Question: {query}

Answer:
"""

    response = llm.invoke(prompt)
    print("\nAnswer:\n", response)