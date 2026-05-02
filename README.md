# 🧠 RAG Under Stress — A Minimal, Reliable RAG System

A lightweight **Retrieval-Augmented Generation (RAG)** pipeline built using LangChain, FAISS, and local/cloud LLMs.
This project focuses not just on building a RAG system, but **analyzing its failure modes and improving reliability**.

---

## 🚀 Features

* 🔍 **Semantic Retrieval** using embeddings (`all-MiniLM-L6-v2`)
* 🧠 **Vector Database** with FAISS (persisted locally)
* ⚡ **Fast Local Inference** via Ollama (`phi3:mini`)
* ☁️ **Optional Cloud Inference** via Hugging Face (Mistral-7B)
* 📊 **Failure Analysis Pipeline** (factual, multi-hop, ambiguous, out-of-scope queries)
* 🛡️ **Strict Grounding** (prevents hallucination with controlled prompting)
* ⚙️ **Optimized for Speed** (persistent index + small chunks)

---

## 📁 Project Structure

```
RAG Under Stress/
├── data/              # Input documents (.txt files)
├── faiss_index/       # Saved vector database
├── rag.py             # Main pipeline
├── results.csv        # Evaluation results (Phase 2)
├── .env               # API keys (optional, for HF)
└── README.md
```

---

## 🧠 How It Works

1. **Load Documents**
   Text files are loaded from the `data/` directory.

2. **Chunking**
   Documents are split into small chunks (350 chars, 50 overlap).

3. **Embeddings**
   Each chunk is converted into vector embeddings.

4. **Vector Storage**
   Stored in a FAISS index for fast similarity search.

5. **Query Flow**

   * User query → embedding
   * Retrieve top-k relevant chunks
   * Pass context + query to LLM
   * Generate grounded response

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install langchain langchain-community langchain-text-splitters \
            langchain-huggingface langchain-ollama \
            sentence-transformers faiss-cpu python-dotenv
```

---

### 2. Add documents

Create a `data/` folder and add 3–5 `.txt` files on a focused topic.

---

### 3. Run with Ollama (local)

Install Ollama and pull model:

```bash
ollama pull phi3:mini
```

Run:

```bash
python rag.py
```

---

### 4. (Optional) Use Hugging Face

Create `.env`:

```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

Switch backend in code:

```python
LLM_BACKEND = "hf"
```

---

## 🧪 Evaluation (Phase 2)

The system was tested across four query types:

| Type         | Description                        |
| ------------ | ---------------------------------- |
| Factual      | Direct knowledge retrieval         |
| Multi-hop    | Requires combining multiple chunks |
| Ambiguous    | Vague queries                      |
| Out-of-scope | Not present in dataset             |

Results are stored in `results.csv`.

---

## 📊 Key Findings

* ✅ Strong performance on factual and multi-hop queries
* ⚠️ Ambiguous queries initially produced verbose answers
* ❌ Early versions showed minor hallucination on out-of-scope queries

---

## 🔧 Improvements (Phase 3)

* Reduced chunk size for better retrieval
* Tuned `k` for optimal context
* Added strict prompt constraints
* Enforced "I don't know" fallback
* Improved response conciseness

---

## 📈 Results

```
Baseline Accuracy: ~7/10
Improved Accuracy: ~10/10
```

---

## ⚠️ Limitations

* Sensitive to vague queries
* No advanced re-ranking
* Basic prompt-based grounding (no guardrails)
* Limited dataset size

---

## 🚀 Future Work

* Add reranking (cross-encoder)
* Better query understanding
* UI for interaction
* Evaluation metrics (precision@k)
* Prompt injection handling

---

## 🧠 What This Project Demonstrates

* End-to-end RAG pipeline design
* Understanding of embeddings + retrieval
* Failure analysis of LLM systems
* Practical ML system optimization

---

## 👤 Author

Built as part of an ML systems exploration project.

---

## ⭐ If you found this useful

Give it a star or fork — and break it further 😉
