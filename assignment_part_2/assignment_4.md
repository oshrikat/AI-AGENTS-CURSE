# Assignment — (agent_s4.py): Conditional Routing

## Compact Overview
This version of the agent (see agents_plots/agent_s4.png) that you are supposed to create, implements an interactive investment assistant as a LangGraph state machine that uses **conditional routing** to decide what to do next based on the user’s message and the current state. It first classifies the user’s intent (explain specific stock(s) vs. suggest a portfolio vs. unclear), then routes to different node paths accordingly. In the portfolio path, it conditionally loops to ask follow-up questions until required profile fields (risk, horizon, budget) are provided, after which it proceeds. In both main branches, it retrieves relevant context from a vector store and then generates the final response, printing a debug trace of the nodes visited so you can see the exact route taken. 

## Interaction Flow
To understand the desired information flow read the sections "Usage of Core Technology" and "Conditional Edges Explanation" from [explanation_agent_s4.md]

## Task
Implement the agent making sure that it contains all the nodes from the diagram.

## some classes that the agent should use
```
from pydantic import BaseModel
class Profile(BaseModel):
    risk_level: Optional[RiskLevel] = None
    investment_horizon_years: Optional[int] = None
    budget_usd: Optional[float] = None
```
## Explanatory Comments on some nodes 
```
def ask_clarification_node(state: AgentState) -> AgentState:
    # Role: When some profile fields are missing, asks the user a single concise question
    # requesting all missing fields (risk level, horizon, budget) and appends the reply
    # to conversation_history so CheckRequirements can re-run on it.
    
def ask_rephrase_node(state: AgentState) -> AgentState:
    # Role: When the intent is "unknown", explains to the user what the agent can do
    # (explain a stock or suggest a portfolio), asks them to rephrase accordingly,
    # and appends the new message so the IntentClassifier can run again.
    
def rag_stock_node(state: AgentState) -> AgentState:
    # Role: For "explain_stock" requests, queries the Chroma vector store using
    # the mentioned tickers (or the raw question) to retrieve the most relevant
    # stock information chunks into state.retrieved_chunks.

def rag_portfolio_node(state: AgentState) -> AgentState:
    # Role: For "suggest_portfolio" requests with a complete profile, builds a query
    # from the user's risk level, horizon, and budget, retrieves matching stock/sector
    # info from the vector store, and stores it in state.retrieved_chunks.
```

## Test Scenarios
Perform the following test scenarious. Save all tests to a docx file (for each test, a record of user-agent interaction). Name of file: tests_[assignment_nr].docx
(e.g. tests_3.docx)
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





