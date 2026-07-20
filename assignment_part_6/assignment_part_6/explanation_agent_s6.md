# Compact Overview

This version of the investment agent (see agents_plots/agent_s6.png) demonstrates **agentic tool use**, where the agent dynamically decides which tools to invoke at each stage of the reasoning process rather than following a hard-coded pipeline. After classifying the user’s intent, the agent enters a tool-calling loop in which it can selectively retrieve stock profiles from a vectorstore, generate recent news snippets, fetch historical risk/return statistics, and compute portfolio-level metrics. Tool outputs are fed back into the conversation as structured messages, allowing the agent to adapt its next action based on newly acquired information. The flow continues until the agent determines it has sufficient evidence to produce a final, schema-constrained JSON response for either stock explanation or portfolio suggestion. This design highlights how decision-making, information gathering, and synthesis are delegated to the agent itself, illustrating a core pattern of modern agentic systems.

# Explanation of the Core Technology (Agentic Tool Use)

Agentic tool use is a paradigm in which a language model is not limited to producing text responses, but instead acts as a decision-making **agent that can intelligently choose actions** during its reasoning process. Rather than following a predetermined sequence of steps, the agent evaluates the current context, identifies what information is missing, and selects the most appropriate tool to obtain that information. This shifts the model’s role from a passive responder to an active controller of its own problem-solving workflow.

At the heart of this approach is the idea that complex tasks are best solved through **iterative interaction** with external capabilities. Tools may represent data sources, computational functions, retrieval systems, or analytical modules. The agent reasons about *which* tool to use, *when* to use it, and *how* to integrate the returned results into its evolving understanding of the task. Importantly, the agent can make multiple tool calls in sequence, adjusting its strategy based on intermediate outputs rather than committing to a single plan upfront.

Another key aspect of agentic tool use is **self-correction and adaptability**. When a tool returns incomplete, unexpected, or invalid results, the agent does not fail immediately. Instead, it incorporates this feedback into its reasoning and selects alternative actions, demonstrating a form of robustness that resembles human problem solving. This enables the system to handle ambiguity, partial information, and constraints dynamically.

Structured outputs play a crucial supporting role in this technology. By requiring the agent to produce final answers that conform to explicit schemas, the system ensures clarity, consistency, and downstream usability of results. The agent must therefore not only gather information, but also synthesize it into a well-defined structure, balancing flexibility during exploration with rigor at the point of conclusion.

Overall, agentic tool use represents a move toward more autonomous and modular AI systems. It enables language models to orchestrate external capabilities intelligently, making them suitable for complex, multi-step tasks such as analysis, planning, and decision support. This technology forms a foundation for advanced agents that can reason, act, and adapt within rich computational environments.

# Usage of the Core Technology (Agentic Tool Use)

This program applies **agentic tool use** by giving the LLM a set of callable tools and then letting it decide, step by step, which tool to invoke to complete the user’s request. The main place this happens is the dedicated advisor node `tool_using_advisor_node`, which binds tools to the model using `llm.bind_tools(tools)` and then runs an iterative loop that keeps calling the LLM until it stops requesting tools.

## Major agent nodes (graph flow)

- `user_input_node` collects the user query and appends it to state via `state.conversation_history.append(msg)`.
- `intent_classifier_node` uses structured output to route the request using `llm.with_structured_output(IntentResult)`, then stores results in `state.intent` and `state.stock_tickers`.
- `collect_portfolio_specs_node` is specific to this version and ensures portfolio requests have the required preferences before tool use, looping until `state.profile.risk_level` and `state.profile.investment_horizon_years` are filled.
- `tool_using_advisor_node` is the core “agentic tool use” node: it decides which tools to call, executes them, and synthesizes the final structured JSON.
- `final_node` prints `state.final_answer` and the debug trace.
- Final answers are expected to be schema-shaped JSON (e.g., `StockExplanation` or `PortfolioSuggestion`) and are stored in state as `state.stock_explanation = StockExplanation(**data)` or `state.portfolio_suggestion = PortfolioSuggestion(**data)`


## Tool exposure and tool-calling loop

Tools are declared as LLM-callable with the decorator `@tool`, for example `@tool def get_stock_profile(ticker: str) -> StockProfile:` and similarly `get_recent_news`, `get_historical_stats`, and `compute_portfolio_metrics`. The advisor exposes them to the model with:

- `tools = [get_stock_profile, get_recent_news, get_historical_stats, compute_portfolio_metrics]`
- `llm_with_tools = llm.bind_tools(tools)`
- `tool_map = {t.name: t for t in tools}`

Inside the loop, the LLM response is checked via `tool_calls = getattr(ai_msg, "tool_calls", None)`. When tool calls exist, the program executes them using `result = tool.invoke(tc["args"])` and feeds results back as `ToolMessage(content=result.model_dump_json(), tool_call_id=tc["id"])`. Errors are also surfaced back into the loop using `ToolMessage(content=f"ERROR: {repr(e)}", tool_call_id=tc["id"])`, enabling the LLM to adapt rather than crash.

## Constraints that shape agent behavior

- The tool_using_advisor_node is strongly instructed via `SystemMessage(content=system_text)` to “call tools” and not rely on prior knowledge.
- A “ticker universe” constraint is enforced using `_allowed_tickers_from_file()` / `_is_ticker_allowed(...)`, and the allowed list is provided to the LLM via the `allowed_block` included in the user message.
- Final answers are expected to be schema-shaped JSON (e.g., `StockExplanation` or `PortfolioSuggestion`) and are stored in state as `state.stock_explanation = StockExplanation(**data)` or `state.portfolio_suggestion = PortfolioSuggestion(**data)`.

Together, these elements implement agentic tool use as a controlled loop: the agent *decides actions (tool calls)*, *executes tools*, *updates context with results*, and *terminates with a structured output* when ready.

# Test Scenarios
## Scenario 1: Explain a Stock (Single-Turn, Tool-Backed JSON)

### User Input
> What do you think about NVDA? Give me a quick explanation and the main risks.

### Expected Agent Behavior
- The agent classifies the intent as **explain_stock**.
- The agent identifies the ticker `NVDA`.
- The agent invokes one or more stock-related tools, such as:
  - get_stock_profile
  - get_historical_stats
- The agent produces a final response that:
  - strictly follows the **StockExplanation** structured schema
  - is returned as JSON only (no free text)
  - includes an educational disclaimer

### Expected Response (Abstract Shape)
json
{
  "intent": "explain_stock",
  "ticker": "NVDA",
  "headline": "...",
  "summary": "...",
  "key_points": ["...", "...", "..."],
  "risk_factors": ["...", "..."],
  "educational_disclaimer": "..."
}


## Scenario 2: Portfolio Request With Missing Specifications (Multi-Turn)

### Turn 1 – User Input
> Build me a portfolio for the next few years. I want something reasonable.

### Expected Agent Behavior (Turn 1)
- The agent classifies the intent as **suggest_portfolio**.
- The agent detects that required portfolio parameters are missing:
  - risk level
  - investment horizon
- The agent responds with a clarification question asking the user to provide these details before proceeding.

---

### Turn 2 – User Input
> Medium risk, 5 years.

### Expected Agent Behavior (Turn 2)
- The agent updates its internal state with:
  - `risk_level = "medium"`
  - `investment_horizon_years = 5`
- The agent selects a small set of suitable stocks based on the provided profile.
- The agent invokes relevant tools, such as:
  - get_stock_profile()
  - get_historical_stats()
  - compute_portfolio_metrics()
- The agent produces a final response that:
  - strictly follows the **PortfolioSuggestion** structured schema
  - explains the rationale for each position at a high level
  - includes an educational disclaimer

### Expected Response (Abstract Shape)
json
{
  "intent": "suggest_portfolio",
  "positions": [
    { "ticker": "AAA", "weight": 0.4, "rationale": "..." },
    { "ticker": "BBB", "weight": 0.35, "rationale": "..." },
    { "ticker": "CCC", "weight": 0.25, "rationale": "..." }
  ],
  "overall_comment": "...",
  "educational_disclaimer": "..."
}

