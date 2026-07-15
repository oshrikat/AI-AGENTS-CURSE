# Code Highlights: LangGraph Agent with Structured Output Extraction

Topic: Using LLM structured output for reliable intent classification and profile extraction in LangGraph agents

---

## Highlight 1: Structured Output Result Models

**Concepts illustrated:**
- Pydantic models define the schema for LLM responses
- `IntentResult` captures intent classification and stock tickers
- `RequirementsResult` captures profile fields and missing field list
- Enables type-safe, directly usable LLM output without parsing

**Code snippet:**

```python
IntentType = Literal["explain_stock", "suggest_portfolio", "unknown"]
class IntentResult(BaseModel):
    intent: IntentType
    stock_tickers: List[str]
    raw_question: str


class RequirementsResult(BaseModel):
    profile: Profile
    missing: List[Literal["risk_level", "investment_horizon_years", "budget_usd"]] = []
```

---

## Highlight 2: Structured Output Intent Classification

**Concepts illustrated:**
- `llm.with_structured_output(Model)` constrains LLM to return typed Pydantic object
- Directly access result fields: `result.intent`, `result.stock_tickers`

**Code snippet:**

```python
def intent_classifier_node(state: AgentState) -> AgentState:
    state = _trace(state, "IntentClassifier")
    latest = state.conversation_history[-1]
    structured_llm = llm.with_structured_output(IntentResult)
    prompt = f"""...
    Fill the fields of IntentResult as follows:
    - intent: one of "explain_stock", "suggest_portfolio", "unknown".
    - stock_tickers: list of tickers mentioned (uppercase), or empty list.
    - raw_question: copy the user message verbatim.
    """
    result: IntentResult = structured_llm.invoke(prompt.strip())

    state.intent = result.intent
    state.stock_tickers = result.stock_tickers
    return state
```

---

## Highlight 3: Structured Output Profile Extraction and Requirements Check

**Concepts illustrated:**
- LLM infers profile fields from user message using structured schema
- Returns list of still-missing fields for clarification loop
- Existing profile values preserved unless contradicted

**Code snippet:**

```python
def check_requirements_node(state: AgentState) -> AgentState:
    state = _trace(state, "CheckRequirements")
    latest = state.conversation_history[-1]
    structured_llm = llm.with_structured_output(RequirementsResult)
    p = state.profile
    prompt = f"""...
    Rules:
    - If the user's message does not contradict existing profile values, keep existing.
    - Only change a field if the user clearly specifies a new value.
    - In "missing", include every field still None after your inference.

    Current profile:
    - risk_level: {p.risk_level}
    - investment_horizon_years: {p.investment_horizon_years}
    - budget_usd: {p.budget_usd}

    User message: \"\"\"{latest}\"\"\""""

    result: RequirementsResult = structured_llm.invoke(prompt.strip())
    state.profile = result.profile
    state.missing_fields = list(result.missing)
    return state
```

---


---