# RAG Chatbot — Document Question Answering with Retrieval-Augmented Generation

## 📌 Objective
Build a **Retrieval-Augmented Generation (RAG)** chatbot that answers questions grounded
strictly in the content of a single source document, rather than relying purely on a language
model's internal (and potentially outdated or hallucinated) knowledge.

## 📄 Source Document
`ml_fundamentals_knowledge_base.txt` — an original ~1,950-word guide covering supervised
learning, unsupervised learning, reinforcement learning, model evaluation, deep learning, and
the end-to-end ML workflow. Written specifically as the knowledge base for this project.

## 🛠️ Tech Stack
- `sentence-transformers` (`all-MiniLM-L6-v2`) — converts text into semantic embedding vectors
- `faiss-cpu` (Facebook AI Similarity Search) — fast vector similarity search / retrieval
- `transformers` (`google/flan-t5-base`) — free, open-source language model for generating
  answers grounded in retrieved context (no paid API key required)
- `numpy` / `pandas` — data handling

## 🔍 Methodology
1. **Document Loading** — loaded the source `.txt` document.
2. **Chunking** — split the document into ~25 paragraph-aligned chunks (~600 characters each),
   preserving coherent ideas rather than breaking mid-sentence.
3. **Embedding** — converted every chunk into a 384-dimensional semantic vector using
   `all-MiniLM-L6-v2`.
4. **Vector Indexing** — built a FAISS `IndexFlatIP` index over the normalized chunk
   embeddings, enabling fast cosine-similarity search.
5. **Retrieval** — given a query, embedded it the same way and retrieved the top-k most
   similar chunks.
6. **Generation** — built a prompt combining the retrieved chunks as context with the user's
   question, explicitly instructing the model to answer only from that context, then generated
   an answer using Flan-T5-base.
7. **Evaluation** — tested the pipeline across multiple in-document questions plus one
   deliberately out-of-scope question, to verify retrieval quality and grounding behavior.
8. **Interactive Chat** — included a runnable input loop for asking free-form questions.

## 📈 Results
Retrieval was verified against the real document: every test query's top-ranked retrieved
chunk was genuinely the most relevant section (e.g. "What is overfitting?" → correctly
retrieves the "Model Evaluation and Overfitting" section as the #1 result). The deliberately
out-of-scope question ("What is the capital of France?") retrieved chunks with a noticeably
lower similarity score than in-scope questions — a useful signal for detecting
knowledge-base coverage gaps.

## ✅ Conclusion
This project built a Retrieval-Augmented Generation (RAG) chatbot that answers questions
grounded strictly in a single source document, using a free, fully open-source stack:
sentence-transformer embeddings for semantic search, a FAISS vector index for fast retrieval,
and a Flan-T5 language model for generating answers constrained to the retrieved context. The
system correctly retrieved relevant passages for in-document questions and showed a clear
similarity-score signal for distinguishing in-scope from out-of-scope questions, demonstrating
the core value proposition of RAG: grounding a language model's answers in a specific,
verifiable, and easily updatable knowledge source rather than relying purely on its internal
training knowledge, which can be outdated, incomplete, or prone to hallucination. A key
limitation of this implementation is the relatively small, free generation model used
(Flan-T5-base), which sometimes produces terse or imperfectly phrased answers compared to what
a larger commercial LLM API would produce — though critically, the retrieval architecture
itself is model-agnostic, so swapping in a more powerful generation model would immediately
improve answer quality without requiring any changes to the retrieval pipeline. This
architecture generalizes directly to much larger, real-world use cases, such as chatbots that
answer questions over an entire company's internal documentation, a legal contract archive, or
a customer support knowledge base.

## 📂 Files
- `RagChatbotCapstone.ipynb` — full notebook with chunking, embedding, retrieval, generation, and evaluation
- `ml_fundamentals_knowledge_base.txt` — the source document the chatbot answers questions from
  (**required** — must be uploaded to Colab alongside the notebook for it to run)
