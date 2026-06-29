# Compact Overview

This program (see agents_plots/agent_s3.png) demonstrates a simple **retrieval-augmented generation (RAG)** investment agent that uses a vector store.  
Instead of hardcoding stock knowledge inside the agent logic, all fictional stock descriptions are stored externally in a local Markdown document and indexed in a vector store.  
The agent first collects user preferences (domain, risk tolerance, and investment horizon), then retrieves the most relevant stock descriptions using semantic similarity search.  
An LLM uses only this retrieved context to construct a concise, preference-aware shortlist of suitable stocks.  

# Explanation of the Core Technology (retrieval-augmented generation)

The core technology demonstrated here is **retrieval-augmented generation (RAG)**, a design pattern that combines information retrieval with large language models to produce grounded, context-aware responses. Instead of relying solely on the model’s internal training data, RAG allows an AI system to consult an external knowledge source at runtime and use the retrieved information as the factual basis for its reasoning and output.

At a high level, RAG works in two stages. First, relevant knowledge is retrieved from a curated corpus based on a semantic query. This corpus can consist of documents, notes, product descriptions, or domain-specific knowledge written by humans. The retrieval step typically relies on vector embeddings, which represent text as numerical vectors capturing semantic meaning. By comparing vectors, the system can identify passages that are conceptually related to a user’s needs, even if they do not share exact keywords.

In the second stage, the retrieved text is provided to a language model as contextual input. The model then generates an answer that is constrained by this external information, rather than inventing details or drawing on unrelated prior knowledge. This makes the system more transparent, easier to control, and better suited for educational or decision-support settings where correctness and traceability matter.

One of the main advantages of RAG is the **separation of knowledge from reasoning**. The language model focuses on interpretation, synthesis, and explanation, while the knowledge itself lives in documents that can be edited, audited, or replaced without retraining the model. This enables rapid iteration, domain customization, and safer experimentation with fictional or hypothetical data.


## Usage of the Core Technology (retrieval-augmented generation)

This program uses **retrieval-augmented generation (RAG)** by inserting a retrieval step between collecting user preferences and asking the LLM to generate recommendations. The result is that the LLM bases its shortlist on **retrieved, local stock descriptions** rather than on memory or assumptions.

### Major nodes (agent steps) in this version
- **PreferenceCollector**: gathers the user’s constraints (domain, risk, horizon) so they can be turned into a retrieval query.
- **StockSelector**: performs semantic retrieval from a persisted vector store and saves the top matching text chunks into state.
- **LLMExplanation**: feeds the retrieved chunks + user preferences to the LLM and asks it to select and justify a shortlist.

### Retrieval (RAG) setup and execution
- The embedding model is created to convert text and queries into vectors:  
  `embeddings = OpenAIEmbeddings(model="text-embedding-3-small")`

- The program opens a **persisted** Chroma vector store (already built earlier from your local stock document):  
  `vectordb = Chroma(persist_directory=VECTORSTORE_DIR, collection_name=COLLECTION_NAME, embedding_function=embeddings)`

- A semantic query is assembled from the user’s preferences (domain/risk/horizon), e.g. via:  
  `query = ", ".join(query_parts) or "general stock info"`

- The actual retrieval step pulls the top-k most similar chunks:  
  `docs = vectordb.similarity_search(query, k=k)`

- The retrieved text is stored in the agent state so it can be used downstream by the generator:  
  `state.retrieved_chunks = [d.page_content for d in docs]`

### Grounded generation using retrieved context
- The program constructs a prompt that includes both the user constraints and the retrieved knowledge blocks:  
  `prompt = build_llm_prompt(state)`

- The LLM is then invoked to choose suitable stocks *only from what was retrieved* and explain the choice:  
  `response = llm.invoke(prompt)`  
  `state.final_answer = response.content`
  


In short: **preferences → semantic retrieval → LLM synthesis**, which is the essential RAG pattern implemented inside an agent graph.

## Test Scenarios
### Scenario 1 — Normal happy path (tech, medium risk, long horizon)

**User input** (interactive CLI):

Domain prompt → tech

Risk prompt → medium

Horizon prompt → long (5y+)

**Expected agent response** (abstract):

Agent prints confirmation that it collected preferences.

Agent prints a DEBUG: Retrieved chunks section with up to 4 chunks that are semantically related to domain: tech, risk level: medium, horizon: long (5y+).

Final answer is a friendly paragraph + bullet points that:

Names up to 3 concrete stocks that appear in the retrieved chunks.

Gives 2–3 sentences per stock explaining why each fits tech + medium risk + long horizon.

Does not recommend stocks that are not mentioned in the retrieved chunks.

### Scenario 2 — “No good match” / cannot identify suitable stocks

**User input** (interactive CLI):

Domain prompt → energy

Risk prompt → low

Horizon prompt → short 0–2y

**Expected agent response** (abstract):

Agent retrieves chunks (still prints the debug preview), but:

Either the chunks are weakly related / off-domain, or

They don’t contain identifiable stock names relevant to the criteria.

Final answer should follow the prompt instruction and explicitly say something equivalent to:
“Sorry, none of the retrieved information matches your criteria well enough.”

No fabricated stock names; if it lists stocks, they must be from retrieved chunks.

### Scenario 3 — Input validation loops (wrong domain and risk first)

**User input** (interactive CLI):

Domain prompt → biotech (invalid)

Domain prompt again → health (valid)

Risk prompt → moderate (invalid)

Risk prompt again → high (valid)

Horizon prompt → medium 2–5y

**Expected agent response** (abstract):

After biotech, agent prints the guidance message:
“choose from tech, health, finance, energy, consumer_goods”

After moderate, agent prints:
“Please type 'low', 'medium', or 'high'.”

Once valid inputs are provided, agent proceeds normally:

Prints debug retrieved chunks (top 4).

Final answer: bullet-point shortlist of up to 3 stocks from retrieved chunks, each justified in 2–3 sentences, aligned to health + high risk + medium horizon.
