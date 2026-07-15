# Compact Overview

This investment assistant (see agents_plots/agent_s5.png) is a CLI-driven LangGraph agent that routes user requests into two main workflows: explaining specific stocks or suggesting a simple three-position portfolio. It illustrates **structured output** by using `llm.with_structured_output(...)` to force the LLM to return typed Pydantic objects for both intent classification (`IntentResult`) and requirements checking (`RequirementsResult`). Those structured results are written into a shared `AgentState`, enabling reliable conditional routing (e.g., ask for clarifications only when required fields are missing) instead of brittle string parsing. The agent also uses a Chroma vector store with OpenAI embeddings to retrieve relevant stock/profile snippets (RAG) before generating the final analysis response.

# Explanation of the Core Technology

**Structured output from large language models** is a technique that constrains an LLM to produce responses in a predefined, machine-readable format rather than free-form text. Instead of asking the model to “answer in words and hoping for consistency,” the developer specifies an explicit schema—often defined using typed data models—describing exactly what fields the response must contain, their types, and their meaning. The LLM then generates an output that conforms to this structure, making its responses predictable, verifiable, and directly usable by downstream systems.

At its core, this approach treats the LLM not just as a text generator, but as a component in a larger software system that must interoperate reliably with code. By enforcing structure, ambiguity is reduced: each response field has a clear semantic role, such as a decision label, a boolean flag, or a list of extracted attributes. This allows the model to participate in logical workflows, decision trees, and state machines without requiring fragile post-processing or heuristic parsing of natural language.

A key benefit of structured output is **robust control flow**. When an LLM’s response is guaranteed to follow a schema, software can branch deterministically based on specific fields, enabling conditional execution paths, validation steps, or follow-up questions. This is especially important in agentic systems, where the model’s output directly influences what happens next rather than being shown only to a human reader.

Another advantage is **error handling and validation**. Structured outputs can be checked automatically: missing fields, invalid values, or schema violations can be detected immediately. This creates a feedback loop where the model can be re-prompted or corrected, improving reliability without manual inspection.

# Usage of core technology (Structured Output)

This program uses **structured output** to turn the LLM into a reliable “decision + extraction” component that drives the agent’s control flow. Instead of parsing free-form text, it asks the model to return **typed objects** (Pydantic models), then routes the graph based on the resulting fields.

## 1) Structured intent classification → deterministic routing
In the `IntentClassifier` node, the LLM is forced to return an `IntentResult` object:

- `structured_llm = llm.with_structured_output(IntentResult)`
- `result: IntentResult = structured_llm.invoke(prompt.strip())`

The agent then updates state from the parsed fields:

- `state.intent = result.intent`
- `state.stock_tickers = result.stock_tickers`

That structured `state.intent` is used by the router:

- `builder.add_conditional_edges("IntentClassifier", route_from_intent, {...})`

So `"explain_stock"` routes to `RAG_Stock`, `"suggest_portfolio"` routes to `CheckRequirements`, and anything else routes to `AskRephrase`.

## 2) Structured requirements extraction → clarification loop
In the `CheckRequirements` node, the LLM is forced to return a `RequirementsResult` object that includes a (partially) completed profile plus an explicit list of missing fields:

- `structured_llm = llm.with_structured_output(RequirementsResult)`
- `result: RequirementsResult = structured_llm.invoke(prompt.strip())`
- `state.profile = result.profile`
- `state.missing_fields = list(result.missing)`

This enables a clean “ask only what’s missing” loop:

- `builder.add_conditional_edges("CheckRequirements", route_from_requirements, {...})`
- If missing fields exist → `AskClarification`
- Otherwise → `RAG_Portfolio`
- After clarification → `builder.add_edge("AskClarification", "CheckRequirements")`



## Test Scenarios
### Scenario 1 — Stock explanation (single-turn)

**User input**:

“What do you think about NVDA? Main risks and outlook please.”

**Expected agent response** (high-level):

Produces a clear explanation of strengths / risks / outlook for NVDA, grounded in retrieved context chunks.

Ends with a short educational disclaimer.

(It does not ask profile questions.)

Expected flow (implicit): IntentClassifier → RAG_Stock → Analysis → Final

### Scenario 2 — Portfolio request with missing fields (clarification loop)

**User input** (turn 1):

“Build me a portfolio of 3 stocks.”

**Expected agent response** (turn 1):

Asks for missing details needed to proceed, e.g.: risk level + horizon + budget (in one message).

**User input** (turn 2):

“Medium risk, 5 years, $10k.”

**Expected agent response** (turn 2, high-level):

Suggests a 3-position portfolio matching the profile.

Gives 2–3 sentences per stock explaining fit.

(No disclaimer is required in the portfolio branch in this version.)

Expected flow (implicit):
IntentClassifier → CheckRequirements → AskClarification → CheckRequirements → RAG_Portfolio → Analysis → Final

### Scenario 3 — Unknown intent → rephrase → stock explanation

**User input** (turn 1):

“Should I invest or just keep cash?”

**Expected agent response** (turn 1):

States it can either explain a specific stock or suggest a portfolio, and asks the user to rephrase accordingly.

**User input** (turn 2):

“Ok—explain TSLA for me.”

**Expected agent response** (turn 2, high-level):

Provides TSLA strengths/risks/outlook using retrieved chunks + ends with an educational disclaimer.

Expected flow (implicit):
IntentClassifier → AskRephrase → IntentClassifier → RAG_Stock → Analysis → Final
