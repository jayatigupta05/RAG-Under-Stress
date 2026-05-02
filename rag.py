from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM # type: ignore
import os
import sys

DATA_DIR = "data"
FAISS_INDEX_PATH = "faiss_index" 
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"                      
OLLAMA_MODEL = "phi3:mini"

# ── 1. Embeddings (loaded once) ───────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},             
)

# ── 2. Load or build FAISS index ─────────────────────────────────────────────
if os.path.exists(FAISS_INDEX_PATH):
    print("Loading existing FAISS index...")
    db = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
else:
    if not os.path.isdir(DATA_DIR):
        print(f"Error: '{DATA_DIR}' folder not found.")        
        sys.exit(1)

    txt_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    if not txt_files:
        print(f"Error: No .txt files found in '{DATA_DIR}'.")
        sys.exit(1)

    print(f"Loading {len(txt_files)} file(s) and building index...")
    docs = []
    for file in txt_files:
        try:
            loader = TextLoader(f"{DATA_DIR}/{file}", encoding="utf-8")
            docs.extend(loader.load())
        except Exception as e:
            print(f"  Warning: skipping {file} — {e}")    

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50,
        add_start_index=True,                                  
    )
    chunks = splitter.split_documents(docs)
    print(f"  Created {len(chunks)} chunks.")

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_INDEX_PATH)                           
    print("  Index saved to disk.")

# ── 3. LLM ───────────────────────────────────────────────────────────────────
llm = OllamaLLM(model=OLLAMA_MODEL)

print("Warming up model...")
try:
    llm.invoke("Hi")
except Exception as e:
    print(f"Error: Could not reach Ollama — is it running? ({e})")
    sys.exit(1)

# ── 4. Query loop ─────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
Answer ONLY using the context below.
If the answer is not there, say "I don't know.
- Use ONLY the provided context
- If the context is empty OR does not contain the answer, respond EXACTLY with: I don't know
- Do NOT use any other knowledge outside of what is provided in the context

Context:
{context}

Question: {query}
Answer:"""

print("\nReady! Type 'exit' to quit.\n")

while True:
    try:
        query = input("Ask something: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    if not query:
        continue
    if query.lower() in {"exit", "quit"}:
        print("Bye!")
        break

    results = db.similarity_search(query, k=4)             

    if not results:
        print("Answer: I don't know (no relevant context found).")
        continue

    context = "\n\n".join(doc.page_content for doc in results)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    print("Answer: ", end="", flush=True)
    for chunk in llm.stream(prompt):                         
        print(chunk, end="", flush=True)
    print()