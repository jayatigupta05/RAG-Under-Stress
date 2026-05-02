from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import os

# 1. Load documents
docs = []
for file in os.listdir("data"):
    if file.endswith(".txt"):
        loader = TextLoader(f"data/{file}")
        docs.extend(loader.load())

# 2. Chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# 3. Embeddings + Vector DB
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.from_documents(chunks, embeddings)

# 4. LLM (Ollama for now)
llm = Ollama(model="phi3:mini")

# 5. Query loop
while True:
    query = input("\nAsk something: ")

    results = db.similarity_search(query, k=4)
    context = "\n\n".join([r.page_content for r in results])

    prompt = f"""
You are a strict assistant.
Answer ONLY using the context below.
If the answer is not there, say "I don't know."

Context:
{context}

Question: {query}
Answer:
"""

    response = llm.invoke(prompt)
    print("\nAnswer:\n", response)