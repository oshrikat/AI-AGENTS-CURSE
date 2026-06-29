## Compact Overview

This program (see agents_plots/agent_s2.png) demonstrates a **simple LangGraph agent with a strictly linear agentic flow**, designed to turn a short CLI interaction into a structured portfolio suggestion. The agent state is defined via `AgentState` and is passed sequentially through the graph using `StateGraph`, preserving user preferences and intermediate results. The flow is explicitly linear—`PreferenceCollector -> StockSelector -> LLMExplanation -> END`—with each node responsible for a single, well-scoped step. Stock selection is handled deterministically via `filter_stocks_by_preferences(...)`, while explanatory reasoning is delegated to an LLM through `llm.invoke(prompt)`. The entire system executes in one pass using `graph.invoke(state)`, illustrating how LangGraph can orchestrate simple agent logic without branching or loops.


## Explanation of the Core Technology

The core technology illustrated here is a **simple LangGraph agent with a linear agentic flow**, which provides a structured way to design and run agents as a sequence of well-defined steps. LangGraph models an agent as a graph of nodes connected by directed edges, where each node represents a distinct unit of reasoning or action. The agent maintains an explicit state that is carried forward from one node to the next, making the flow of information transparent and controllable.

In a linear agentic flow, the execution path is fixed in advance and always follows the same order of nodes. There is no branching, looping, or dynamic decision about which step to execute next. This makes the behavior of the agent predictable, easy to debug, and especially suitable for introductory or educational settings where the goal is to understand the mechanics of agent orchestration rather than to maximize flexibility.

A key advantage of this approach is the clear separation of concerns. Each node can focus on a single responsibility, such as gathering inputs, applying rules, or generating explanations, while LangGraph handles the coordination between these steps. This structure encourages modular thinking and helps developers reason about complex agent behavior by breaking it down into simpler, sequential components.

Overall, a linear LangGraph agent demonstrates how agent-based systems can combine deterministic logic with language-model reasoning in a controlled and interpretable way. It serves as a foundational pattern that can later be extended with conditional branches, loops, memory, or tool use, while still preserving the clarity and discipline of an explicit graph-based execution model.

## Usage of the core technology in this program (implementation of a LangGraph agent)

This program uses LangGraph to implement a **linear agentic flow** by defining a state schema, registering nodes, wiring them in a fixed sequence, compiling the graph, and then invoking it once.

- **State is explicit and shared across nodes** via a Pydantic model: `class AgentState(BaseModel): ...`  
  This state holds the user’s preferences (`domain`, `risk`, `horizon`), intermediate results (`selected_stocks`), and the final output (`final_answer`).

- **A linear LangGraph is constructed** using `builder = StateGraph(AgentState)` and three node registrations:  
  `builder.add_node("PreferenceCollector", preference_collector_node)`  
  `builder.add_node("StockSelector", stock_selector_node)`  
  `builder.add_node("LLMExplanation", llm_explanation_node)`  
  These are the major nodes specific to this version of the agent: `PreferenceCollector`, `StockSelector`, and `LLMExplanation`.

- **The entry point is fixed** with `builder.set_entry_point("PreferenceCollector")`, ensuring the run always starts by collecting user preferences.

- **The linear execution order is enforced** by explicit edges:  
  `builder.add_edge("PreferenceCollector", "StockSelector")`  
  `builder.add_edge("StockSelector", "LLMExplanation")`  
  `builder.add_edge("LLMExplanation", END)`  
  This hard-codes a single path through the graph (no branching or loops).

- **Nodes can route execution using LangGraph Commands.** For example, `stock_selector_node` returns  
  `Command(update=state, goto="LLMExplanation")`  
  and the terminal steps route to `END` via  
  `Command(update=state, goto=END)`.

- **The graph is compiled and executed once**: `return builder.compile()` and later `graph.invoke(state)` inside `run_agent_cli()`.  
  The final node delegates natural-language explanation to the LLM using `response = llm.invoke(prompt)`, then stores it in `state.final_answer`.

## Test Scenarios
### Test scenario 1 — Happy path (enough matches)

**User input** (single run, CLI prompts):

Domain: tech

Risk tolerance: medium

Investment horizon: long (5+ years)

**Expected agent response** (shape/content):

A short intro acknowledging the preferences.

3–5 bullet points, each bullet is a stock ticker from the candidate stocks list that match:

domain=tech and risk_level=medium

For each bullet: 2–3 sentences explaining why it fits a medium-risk, long-horizon tech preference, referencing the stock’s outlook and key_risks (high-level, not necessarily numeric or “actionable”).



### Test scenario 2 — No matches (must trigger the explicit apology message)

**User input** (single run, CLI prompts):

Domain: energy

Risk tolerance: low

Investment horizon: short (0–2 years)

**Expected agent response**:

The response must include the exact sentence (as required by the prompt built in the agent):

Sorry, none of the provided stocks matches your criteria

No stock bullets (or an empty/brief follow-up explanation like “try changing domain or risk” is fine, but the key requirement is that apology line appears).

