# Code Highlights: Simple LangGraph Agent with Linear Flow

Topic: Building a LangGraph agent with explicit state and linear orchestration

---

## Highlight 1: Explicit State Schema

**Concepts illustrated:**
- Pydantic BaseModel for state definition
    -  State holds preferences, intermediates, and final answer 
    - Typed fields 
    - Shared state across all nodes


**Code snippet:**

```python
from pydantic import BaseModel 

class AgentState(BaseModel):
    domain: Optional[str] = None
    risk: Optional[RiskLevel] = None
    horizon: Optional[str] = None
    selected_stocks: List[str] = []
    final_answer: Optional[str] = None
```

---

## Highlight 2: Node Functions Returning State

**Concepts illustrated:**
- Node as a pure function: input state → output state
    - usually, node modifies the state
    - Returning `state` passes control to the next node


**Code snippet:**

```python
def stock_selector_node(state: AgentState):
    selected = filter_stocks_by_preferences(
        stocks=stocks_dict,
        domain=state.domain,
        risk=state.risk
    )
    state.selected_stocks = selected
    return state


def llm_explanation_node(state: AgentState):
    prompt = build_llm_prompt(state)
    response = llm.invoke(prompt)
    state.final_answer = response.content
    print("\nAGENT (final suggestions):\n")
    print(state.final_answer)
    return state
```

---

## Highlight 3: StateGraph Construction

**Concepts illustrated:**
- `StateGraph(AgentState)` initializes graph with state schema
    -   `builder.add_node(name, function)` registers nodes


**Code snippet:**

```python
def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("PreferenceCollector", preference_collector_node)
    builder.add_node("StockSelector", stock_selector_node)

```

---

## Highlight 4: Linear Flow Wiring

**Concepts illustrated:**
- `set_entry_point` marks the starting node
- `add_edge(from, to)` creates directed connections
- Linear flow: PreferenceCollector → StockSelector → LLMExplanation → END
- no branching or loops

**Code snippet:**

```python
    builder.set_entry_point("PreferenceCollector")

    builder.add_edge("PreferenceCollector", "StockSelector")
    builder.add_edge("StockSelector", "LLMExplanation")
    builder.add_edge("LLMExplanation", END)

    agent = builder.compile()
```

---

## Highlight 5: Graph Invocation

**Concepts illustrated:**
- `builder.compile()` produces runnable agent
- `graph.stream(state)` executes agent node-by-node
- Each yielded object (`chunk`) is usually a dictionary containing:
    - the node name
    - the updated state produced by that node

**Code snippet:**

```python
if __name__ == "__main__":
    graph = build_graph()
    state = AgentState()
    for chunk in graph.stream(state):
        print(f"\n--- STEP: {list(chunk.keys())} ---")
        print(chunk)
```

---

