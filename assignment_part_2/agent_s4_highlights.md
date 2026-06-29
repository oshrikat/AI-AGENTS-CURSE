# Code Highlights: LangGraph Agent with Conditional Routing and Clarification Loops

Topic: Building a multi-intent investment agent with intent classification, profile extraction, and conditional graph routing

---

## Highlight 1: Nested State Schema with Profile

**Concepts illustrated:**
- Pydantic BaseModel for type-safe state
- Nested `Profile` model within `AgentState`
- Conversation history for multi-turn dialogue
- Debug trace field for node visitation logging

**Code snippet:**

```python
class Profile(BaseModel):
    risk_level: Optional[RiskLevel] = None
    investment_horizon_years: Optional[int] = None
    budget_usd: Optional[float] = None


class AgentState(BaseModel):
    conversation_history: List[str] = []
    intent: Optional[IntentType] = None
    stock_tickers: List[str] = []
    profile: Profile = Profile()
    missing_fields: List[str] = []
    retrieved_chunks: List[str] = []
    final_answer: Optional[str] = None
    node_trace: List[str] = []
```


---

## Highlight 2: Regex-based Profile Field Extraction

**Concepts illustrated:**
- Heuristic extraction from natural language text
- Risk level detection from keywords
- Investment horizon extraction with regex (e.g., "5 years")
- Budget extraction with suffix handling (k, m)

**Code snippet:**

```python
def update_profile_from_text(text: str, profile: Profile) -> Profile:
    lower = text.lower()

    if profile.risk_level is None:
        if "low" in lower: profile.risk_level = "low"
        elif "medium" in lower: profile.risk_level = "medium"
        elif "high" in lower: profile.risk_level = "high"

    if profile.investment_horizon_years is None:
        m = re.search(r"(\d+)\s*(?:year|years|yr|yrs|y)\b", lower)
        if m:
            profile.investment_horizon_years = int(m.group(1))

    if profile.budget_usd is None:
        m = re.search(r"\$?(\d+(?:\.\d+)?)\s*(k|m|usd|dollars)?", lower)
        if m:
            amount = float(m.group(1))
            suffix = (m.group(2) or "").lower()
            if suffix == "k": amount *= 1_000
            elif suffix == "m": amount *= 1_000_000
            profile.budget_usd = amount

    return profile
```

---

## Highlight 3: Conditional Routing Functions

**Concepts illustrated:**
- Router functions return node name strings
- Used by `add_conditional_edges` to decide next node
- First router: intent-based routing
- Second router: requirements completeness check

**Code snippet:**

```python
def route_from_intent(state: AgentState) -> str:
    if state.intent == "explain_stock":
        return "RAG_Stock"
    elif state.intent == "suggest_portfolio":
        return "CheckRequirements"
    else:
        return "AskRephrase"


def route_from_requirements(state: AgentState) -> str:
    if state.missing_fields:
        return "AskClarification"
    else:
        return "RAG_Portfolio"
```

---

## Highlight 4: Conditional Edges with Loop Backs

**Concepts illustrated:**
- `add_conditional_edges(source, router_func, mapping_dict)` enables branching
- Loop back to `CheckRequirements` after user provides clarification
- Loop back to `IntentClassifier` after user rephrases request
- Enables multi-turn conversation within a single graph invocation

**Code snippet:**

```python
builder.add_conditional_edges(
    "IntentClassifier",
    route_from_intent,
    {
        "RAG_Stock": "RAG_Stock",
        "CheckRequirements": "CheckRequirements",
        "AskRephrase": "AskRephrase",
    },
)

builder.add_conditional_edges(
    "CheckRequirements",
    route_from_requirements,
    {
        "AskClarification": "AskClarification",
        "RAG_Portfolio": "RAG_Portfolio",
    },
)

builder.add_edge("AskClarification", "CheckRequirements")
builder.add_edge("AskRephrase", "IntentClassifier")
```

---