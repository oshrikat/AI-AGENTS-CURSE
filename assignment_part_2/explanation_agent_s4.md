# Compact Overview

This program implements an interactive investment assistant (see agents_plots/agent_s4.png) as a LangGraph state machine that uses **conditional routing** to decide what to do next based on the user’s message and the current state. It first classifies the user’s intent (explain specific stock(s) vs. suggest a portfolio vs. unclear), then routes to different node paths accordingly. In the portfolio path, it conditionally loops to ask follow-up questions until required profile fields (risk, horizon, budget) are provided, after which it proceeds. In both main branches, it retrieves relevant context from a vector store and then generates the final response, printing a debug trace of the nodes visited so you can see the exact route taken. 

# Explanation of the Core Technology (Conditional Routing)

Conditional routing is a control paradigm used in agentic and workflow-based AI systems to dynamically determine the next step in a process based on context, state, or observed signals. Instead of executing a fixed, linear sequence of operations, a system built around conditional routing evaluates conditions at runtime and selectively activates different paths of execution. This allows the system to adapt its behavior to user intent, missing information, or evolving goals, making interactions more flexible and context-aware.

At its core, conditional routing relies on three elements: **state**, **conditions**, and **transitions**. The state represents everything the system currently knows, such as user inputs, accumulated constraints, or intermediate decisions. Conditions are logical checks applied to that state—for example, whether sufficient information has been provided, or whether the user’s request matches a known category. Transitions define where the system should go next when a condition is met, effectively forming a directed graph of possible execution paths.

One of the main advantages of conditional routing is its support for iterative and non-linear interaction patterns. A system can pause progress, request clarification, and then resume from exactly the right point once new information is available. This contrasts with traditional pipelines, where missing or ambiguous inputs often cause failure or require restarting the entire process. With conditional routing, loops, branches, and early exits are first-class design features rather than exceptions.

Conditional routing is especially important in AI-driven assistants, where user intent is rarely known upfront and may shift during the interaction. By separating *decision-making* (which route to take) from *capability execution* (what each route does), the system becomes easier to extend and reason about. New behaviors can be added by introducing new conditions and routes without rewriting existing logic.

# Usage of Core Technology (Conditional Routing)

In this program, **conditional routing** is realized by structuring the agent as a directed graph of nodes and explicitly defining decision points that choose the next node based on the current state. Instead of a single linear flow, execution paths diverge and sometimes loop, depending on what the system infers about the user’s request and what information is still missing.

A first major use of conditional routing appears immediately after user input. The program classifies the intent and routes accordingly using logic of the form  
`add_conditional_edges(...)`.  
Here, the output of an intent-classification step determines whether the agent proceeds toward explaining specific stocks, toward building a portfolio, or toward a fallback path for unclear requests. This illustrates routing based on *semantic interpretation* rather than simple command matching.

In the portfolio-building path, conditional routing is used iteratively. A dedicated node (e.g. `CollectPortfolioSpecs`) checks whether required attributes such as risk level or investment horizon are already present in the state. If not, routing sends execution back to a question-asking node instead of moving forward. Conceptually, this is implemented with checks like  
`if not state.risk_level: return "ask_risk"`  
which cause the graph to loop until all conditions are satisfied. Only then does routing advance to the portfolio construction logic.

Another important routing point separates *information gathering* from *response generation*. After routing decides that enough context exists, the agent transitions to nodes responsible for retrieval and synthesis (RAG_Stock_node, RAG_portfolio_node), eventually ending in a terminal node such as `Final`. This makes completion conditional on state completeness rather than on a fixed number of steps.

Key nodes that embody this approach in this version of the agent include `IntentClassifier`, `CollectPortfolioSpecs`, `ExplainStocks`, and `Final`. Together, they demonstrate how conditional routing turns a static assistant into a state-aware system that can branch, pause, and resume intelligently based on evolving interaction context.

# Detailed Documentation 
## Core Capabilities
    • Intent Detection:
The agent classifies each incoming message into one of three intents:
explain_stock, suggest_portfolio, or unknown.
This determines which branch of the graph the agent will follow.
    • Dynamic Profiling:
For portfolio requests, the agent attempts to automatically extract:
        ◦ risk level
        ◦ investment horizon (years)
        ◦ budget (USD)
If any of these details are missing, the agent asks the user concise clarification questions until the profile is complete.
    • Retrieval-Augmented Generation (RAG):
The agent searches a Chroma vector store of stock profiles to gather the most relevant chunks of information.
Two RAG paths exist:
        ◦ RAG_Stock (for stock explanation)
        ◦ RAG_Portfolio (for portfolio construction)
    • LLM Analysis:
Once the appropriate data is retrieved, the agent generates either:
        ◦ a stock explanation (strengths, risks, outlook), or
        ◦ a simplified portfolio recommendation aligned with risk level, horizon, and budget.
## Graph Structure
The agent is built as a branching LangGraph state machine, with the following key nodes:
    1. UserInput – collects the user’s query
    2. IntentClassifier – decides what task is required
    3. CheckRequirements – checks if portfolio info is complete
    4. AskClarification – asks for missing profile fields
    5. AskRephrase – guides the user when intent is unclear
    6. RAG_Stock / RAG_Portfolio – retrieve relevant textual chunks
    7. Analysis – produces final natural-language output
    8. Final – prints the answer and the node path
## Routing Logic
    • explain_stock → RAG_Stock → Analysis → Final
    • suggest_portfolio → CheckRequirements →
        ◦ missing fields → AskClarification → CheckRequirements
        ◦ complete → RAG_Portfolio → Analysis → Final
    • unknown → AskRephrase → IntentClassifier
## Purpose
This stage demonstrates how to build a truly agentic system that:
    • interprets ambiguous user requests,
    • resolves missing information through multi-turn dialogue,
    • branches into different reasoning paths,
    • integrates RAG for data grounding,
    • and orchestrates the entire flow through structured node routing.

## Conditional Edges Explanation
-------------------------------

1. Conditional edges from IntentClassifier

These edges decide which branch the agent follows based on state.intent.

If intent == "explain_stock" → go to RAG_Stock.
If intent == "suggest_portfolio" → go to CheckRequirements.
If intent == "unknown" → go to AskRephrase.

This chooses between stock explanation, portfolio suggestion, or asking the user to rephrase.

2. Conditional edges from CheckRequirements

These edges decide whether the profile is complete.

If missing_fields is non-empty → go to AskClarification.
If missing_fields is empty → go to RAG_Portfolio.

This ensures the agent asks for missing details before making a portfolio.

3. Loops Created Around These Edges

AskClarification always returns to CheckRequirements after user input.
AskRephrase always returns to IntentClassifier.

This creates loops:

CheckRequirements → AskClarification → CheckRequirements → ...
IntentClassifier → AskRephrase → IntentClassifier → ...

4. Full Routing Picture

UserInput
  ↓
IntentClassifier --(explain_stock)--> RAG_Stock → Analysis → Final
       │
       ├--(suggest_portfolio)--> CheckRequirements
       │                         ├--(missing_fields)--> AskClarification → CheckRequirements
       │                         └--(complete)--> RAG_Portfolio → Analysis → Final
       │
       └--(unknown)--> AskRephrase → IntentClassifier



## Test Scenarios
### Scenario 1 — Explain a specific stock (single-turn)

**User input**

“What do you think about NVDA? Main risks and upside?”

**Expected agent response** (abstract)

The agent classifies intent as explain_stock, extracts ticker NVDA, retrieves a few relevant chunks from the vector store, and returns:

A short structured explanation: strengths, risks, outlook.

A brief educational disclaimer at the end (not financial advice).

(Optional debug expectation) Node path resembles: UserInput -> IntentClassifier -> RAG_Stock -> Analysis -> Final.

### Scenario 2 — Portfolio request with missing fields (multi-turn clarification loop)

Dialogue

**User input**

“Build me a portfolio.”

**Expected agent response**

The agent classifies suggest_portfolio, sees missing risk_level, investment_horizon_years, budget_usd, and asks a single clarification question requesting all missing fields in one message.

**User input**

“Medium risk, 5 years, $10k.”

**Expected agent response** (abstract)

The agent now has a complete profile, runs portfolio RAG, and returns:

A 3-position stock portfolio suggestion aligned with medium risk / 5y / $10k.

For each position: 2–3 sentences “why it fits”.

(Optional debug expectation) Node path includes the loop:
... -> CheckRequirements -> AskClarification -> CheckRequirements -> RAG_Portfolio -> Analysis -> Final.

### Scenario 3 — Unclear intent → rephrase loop → then stock explanation (multi-turn)

Dialogue

**User input**

“Is now a good time to invest?”

**Expected agent response**

The agent classifies unknown and asks the user to rephrase into one of:

“explain a specific stock”, or

“suggest a portfolio”.

**User input**

“Explain AAPL and TSLA briefly.”

**Expected agent response** (abstract)

The agent classifies explain_stock, extracts tickers AAPL, TSLA, retrieves chunks, and returns:

A concise per-stock explanation (strengths/risks/outlook), and a disclaimer.

(Optional debug expectation) Node path includes the rephrase loop:
UserInput -> IntentClassifier -> AskRephrase -> IntentClassifier -> RAG_Stock -> Analysis -> Final.


