# Assignment — Stage 2: Agentic version of Stage 1 interaction flow

## Compact Overview

This program (see agents_plots/agent_s2.png) should demonstrate a **simple LangGraph agent with a strictly linear agentic flow**, designed to turn a short CLI interaction into a structured portfolio suggestion. The agent state is defined via `AgentState` and is passed sequentially through the graph using `StateGraph`, preserving user preferences and intermediate results. The flow is explicitly linear—`PreferenceCollector -> StockSelector -> LLMExplanation -> END`—with each node responsible for a single, well-scoped step. Stock selection is handled deterministically via `filter_stocks_by_preferences(...)`, while explanatory reasoning is delegated to an LLM through `llm.invoke(prompt)`. The entire system executes in one pass using `graph.invoke(state)`, illustrating how LangGraph can orchestrate simple agent logic without branching or loops.


## Task
Same interaction flow as in Stage 1 but implemented as an agent. A simple LangGraph agent with linear interaction flow (no loops, no conditions).
Plot diagram of the agent.

## Agent state
It is recommended to use the class AgentState (as the agentic state)
```
from pydantic import BaseModel
class AgentState(BaseModel):
    domain: Optional[str] = None
    risk: Optional[RiskLevel] = None
    horizon: Optional[str] = None  # e.g. "short", "medium", "long"
    selected_stocks: List[str] = []
    final_answer: Optional[str] = None
```

## Required Nodes

Implement **the following functions/nodes**, with the same names and roles as in the provided code.


## 1. preference_collector_node
```
def preference_collector_node(state: AgentState) -> AgentState:
```
has same function as ask_user_preferences() from previous stage.


## 2. stock_selector_node
```
def stock_selector_node(state: AgentState):-> AgentState:
```
has same function as filter_stocks_by_preferences() from previous stage.



## 3. def llm_explanation_node
```
def llm_explanation_node(state: AgentState):->AgentState
```
has same function as generate_explanations_with_llm() from previous stage

## some other functions (not nodes)
### build graph
```
def build_graph()
```
Function that builds the Graph (which is basically the agent) that consists of nodes and edges and returns it.

## main function
in the solution (that uses langgraph 1.0.4) main function looks like this.
```
if __name__ == "__main__":
    graph = build_graph()
    state = AgentState()
    # Single linear run: all interaction happens inside the nodes (via input / print)
    graph.invoke(state)

```

## Plotting the Diagram of the Agent
insert a code snippet that creates a diagram of the agent and to do the same for all subsequent agents. here is how it looks like in my program (langgraph 1.0.4)(I put it in build_graph())
```
...
builder.add_edge("Final", END)
agent = builder.compile()
if plot_graph:
    png_bytes = agent.get_graph().draw_mermaid_png()
    with open("agent_s2.png", "wb") as f:
        f.write(png_bytes)
```



## Comments
- no need to write all the code inside the nodes. you can write functions that will be used in the nodes.

- feel free to code (functions etc.) from previous stages. maybe, best practice is to locate functions that are used in multiple programs in separate file(s) such as utils.py.

- I recommend that you should have the function build_graph() in all versions of the agent although it is not explicitely required in each assignment. the purpose of this function is to merge all the nodes into a graph.

- I recommend to use class AgentState in more advanced versions of the agent as well and to include more variables into it if required








