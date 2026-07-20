# Code Highlights: Agentic Tool Use in LangGraph Agents

Topic: Building a LangGraph agent that dynamically decides which tools to invoke during reasoning, implementing an iterative tool-calling loop

---

## Highlight 1: LangChain Tool Definitions

**Concepts illustrated:**
- `@tool` decorator marks functions as LLM-callable
- such functions are used as tools
**Code snippet:**

```python
@tool
def get_stock_profile(ticker: str) -> StockProfile:
    """Return a stock profile grounded in the vectorstore text."""
    ...
    return profile


@tool
def compute_portfolio_metrics(positions: List[PositionInput]) -> PortfolioMetrics:
    """Compute mock portfolio metrics from positions."""
    ...
```

---

## Highlight 2: Tool Binding and Tool Map

**Concepts illustrated:**
- `llm.bind_tools(tools)` enables tool-calling mode on the LLM
- `tool_map` dictionary maps tool names to tool objects for lookup
- LLM decides which tools to call based on context

**Code snippet:**

```python
tools = [
    get_stock_profile,
    get_recent_news,
    get_historical_stats,
    compute_portfolio_metrics,
]
llm_with_tools = llm.bind_tools(tools)
tool_map = {t.name: t for t in tools}
```

---


---

## Highlight 3: System Prompt for Tool-First Behavior

**Concepts illustrated:**
- System message instructs LLM to use tools instead of prior knowledge
- Explicit rules: must call at least one tool, never hallucinate
- Schema templates define expected output structure

**Code snippet:** (from Tool_using_advisor_node)

```python
system_text = (
    "You are an investment assistant that MUST call tools to answer.\n"
    "You have access to the following tools:\n"
    "- get_stock_profile(ticker)\n"
    "- get_recent_news(query)\n"
    "- get_historical_stats(ticker)\n"
    "- compute_portfolio_metrics(positions)\n\n"
    "Rules:\n"
    "1. You are NOT allowed to answer using your own prior knowledge.\n"
    "2. ALL factual details in your final JSON MUST come from tool results.\n"
    "3. You MUST make at least one tool call before producing the final answer.\n"
    "4. If you are missing information, you MUST call additional tools instead of guessing.\n"
    "5. Never hallucinate or invent numbers, news, or profiles.\n"
    "6. If a tool response contains an ERROR saying a ticker is not available, "
    "   you MUST choose a different ticker from the allowed universe and try again.\n\n"
    ...
)
```

---
## Highlight 4: Tool-Calling Loop

**Comments on snippet  ** 
- `ai_msg = llm_with_tools.invoke(messages)`
	- messages contains user prompt (Human Message)(e.g. "Tell me about NVDA") as well as other messages, e.g.
	messages = [  
SystemMessage(content=system_text),  
HumanMessage(content=user_input),  
  
AIMessage(tool_calls=[...]),  
ToolMessage(content=tool_result),  
  
AIMessage(tool_calls=[...]),  
ToolMessage(content=another_result),  
]
	- each pair (AIMessage,ToolMessage) corresponds to 1 iteration of tool calling loop (the while loop)

**Concepts illustrated:**
- While loop executes tools until LLM decides that tool calls are not required any more
- each iteration may call 1 or multiple tools
- ai_msg (response from llm_with_tools) contains fields "content" and "tool_calls"
	- "tool calls" contains list of suggested tools as well as arguments for each tool
- `tool.invoke(tc["args"])` executes the tool with provided arguments
- `ToolMessage` appends results back to conversation history

**Code snippet:**

```python
        max_tool_calls = 10
        has_pending_tool_calls = True
        tc_iteration = 0
        final_ai_msg = None

        while has_pending_tool_calls and tc_iteration < max_tool_calls:
            tc_iteration += 1

            ai_msg = llm_with_tools.invoke(messages)
            tool_calls = getattr(ai_msg, "tool_calls", None)

            has_pending_tool_calls = bool(tool_calls)

            if not has_pending_tool_calls:
                final_ai_msg = ai_msg

            messages.append(ai_msg)

            for tc in tool_calls:
                tool = tool_map.get(tc["name"])

                if tool is None:
                    messages.append(
                        ToolMessage(
                            content=f"ERROR: Unknown tool {tc['name']}",
                            tool_call_id=tc["id"],
                        )
                    )
                    continue

                try:
                    result = tool.invoke(tc["args"])

                    if hasattr(result, "model_dump_json"):
                        content = result.model_dump_json()
                    else:
                        content = json.dumps(result)

                except Exception as e:
                    content = f"ERROR: {repr(e)}"

                messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=tc["id"],
                    )
                )

        if final_ai_msg is None:
            state.final_answer = (
                "ToolUsingAdvisor stopped because it reached the maximum number "
                f"of tool-calling iterations ({max_iterations}) without producing "
                "a final answer."
            )
            return state

        # -----------------------------
        # Final JSON parsing
        # -----------------------------
        raw = final_ai_msg.content
        data = json.loads(raw)
        ...

```
