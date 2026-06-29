# Code Highlights: LangGraph Agent with RAG Pipeline

Topic: Building a LangGraph agent that retrieves relevant stock information from a vector database

---

## Highlight 1: Creation of Vector Store

**Concepts illustrated:**

- Creation of Vector Store
- Chroma Vector Store
- uses embedding model (e.g. "text-embedding-3-small") to create embedding of a text chunk

### Example of s["content"] (for following code snippet)

```


CVX
- **Domain:** Energy
- **Risk level:** medium
- **Outlook:** Positive near-term outlook backed by strong balance sheet and global demand.
- **Key risks:** Commodity price swings; Geopolitical exposure; Energy-transition pressure

```

**Code snippet:** (see [[profiles_to_vecstore.py]])

```python
from langchain_chroma import Chroma
...
emb = "text-embedding-3-small"
texts = [s["content"] for s in sections if s["content"]]
metadatas = [{"ticker": s["title"]} for s in sections if s["content"]]

embedding = OpenAIEmbeddings(model=emb)
Chroma.from_texts(
    texts=texts,
    metadatas=metadatas,
    embedding=embedding,
    persist_directory="chroma_store",
    collection_name="stock_profiles"
)
```

---

---

## Highlight 2: Stock Selector Node with RAG Logic

**Concepts illustrated:**

- Build query string from user preferences (domain, risk, horizon)
- Load Chroma vector store with embedding function
- `similarity_search(query, k=4)` retrieves top-k relevant chunks
- Store retrieved chunks in state for downstream use

**Code snippet:**

```python
def stock_selector_node(state: AgentState):
    query_parts = []
    if state.domain:
        query_parts.append(f"domain: {state.domain}")
    if state.risk:
        query_parts.append(f"risk level: {state.risk}")
    if state.horizon:
        query_parts.append(f"horizon: {state.horizon}")
    query = ", ".join(query_parts) or "general stock info"

    vectordb = Chroma(
        persist_directory=VECTORSTORE_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )
    docs = vectordb.similarity_search(query, k=k)
    state.retrieved_chunks = [d.page_content for d in docs]

    print("\nDEBUG: Retrieved chunks:")
    for i, chunk in enumerate(state.retrieved_chunks, start=1):
        print("  Chunk {}: {}...".format(i, chunk[:200].replace("\n", " ")))
    return state
```

---

## Highlight 3: LLM Prompt Built from Retrieved Chunks

**Concepts illustrated:**

- Convert list of retrieved chunks into readable text
- Inject chunks as context into LLM prompt
- Task: identify stocks from chunks and explain fit to user preferences

**Code snippet:**

```python
def build_llm_prompt(state: AgentState) -> str:
    if state.retrieved_chunks:
        chunks_text = "\n\n".join(
            f"Chunk {i+1}:\n{chunk}"
            for i, chunk in enumerate(state.retrieved_chunks)
        )
    else:
        chunks_text = "No relevant information was retrieved."

    prompt = f"""
You are an educational investment assistant.

User preferences:
- Domain: {state.domain}
- Risk tolerance: {state.risk}
- Investment horizon: {state.horizon}

Retrieved knowledge:
{chunks_text}

Task:
1. Identify concrete stocks mentioned in the retrieved chunks.
2. Select the most suitable 3 stocks (or fewer if not enough information).
3. For each chosen stock, give a 2–3 sentence explanation of why it fits.
"""
    return prompt.strip()
```

---

---