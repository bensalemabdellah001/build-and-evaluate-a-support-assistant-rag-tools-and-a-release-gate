# Build and Evaluate a Support Assistant: RAG, Tools and a Release Gate

Build the support assistant for an outdoor gear store from its help center, the way an LLM product team does. Chunk the pages so every chunk keeps its heading, retrieve with keyword and embedding search fused together, and assemble a prompt under a token budget. Parse and validate the model's JSON, repair it when it breaks, and send hard questions from a cheap model to a strong one. Add tools behind a permission gate, reuse answers with a semantic cache, and evaluate every version on 20 test questions. Then watch a release gate block the version that follows an instruction planted on a forum page, and ship the one that holds it. The models and embeddings are deterministic stand-ins, so every step is exactly testable.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** chunk_document
- [x] **2.** bm25_scores
- [x] **3.** hybrid_search
- [x] **4.** build_prompt
- [x] **5.** parse_answer
- [x] **6.** ask
- [x] **7.** cascade
- [ ] **8.** permission_gate
- [ ] **9.** run_turn
- [ ] **10.** SemanticCache
- [ ] **11.** evaluate
- [ ] **12.** release_gate

---

Built on Deep-ML.
