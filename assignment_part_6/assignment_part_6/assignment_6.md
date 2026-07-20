# Assignment — Stage 6: Agentic Tool Use
## Compact Overview
This version of the investment consultant (see agents_plots/agent_s6.png) demonstrates **agentic tool use**, where the agent dynamically decides which tools to invoke at each stage of the reasoning process rather than following a hard-coded pipeline. After classifying the user’s intent, the agent enters a tool-calling loop in which he can selectively retrieve stock profiles from a vectorstore, generate recent news snippets, fetch historical risk/return statistics, and compute portfolio-level metrics. Tool outputs are fed back into the conversation as structured messages, allowing the agent to adapt its next action based on newly acquired information. The flow continues until the agent determines it has sufficient evidence to produce a final, schema-constrained JSON response for either stock explanation or portfolio suggestion. This design highlights how decision-making, information gathering, and synthesis are delegated to the agent itself, illustrating a core pattern of modern agentic systems.

## Task
Implement the agent in accordance with the plot and the section **Usage of the Core Technology** from [explanation_agent_s6.md]

## Required Classes
```
RiskLevel = Literal["low", "medium", "high"]
IntentType = Literal["explain_stock", "suggest_portfolio", "unknown"]

class Profile(BaseModel):
    # Stores the core user investment preferences used for portfolio construction
    risk_level: Optional[RiskLevel] = None
    investment_horizon_years: Optional[int] = None
    budget_usd: Optional[float] = None


class IntentResult(BaseModel):
    # Structured result for the intent classifier node
    intent: IntentType
    stock_tickers: List[str]
    raw_question: str

# --- new tool data models + structured outputs (place after IntentResult / RequirementsResult) ---

class StockProfile(BaseModel):
    """Simple data container for a stock profile returned by get_stock_profile."""
    ticker: str
    name: str
    sector: str
    description: str
    risk_level: RiskLevel


class NewsSnippet(BaseModel):
    """One mock news snippet."""
    title: str
    snippet: str


class NewsResult(BaseModel):
    """Container for multiple mock news snippets."""
    items: List[NewsSnippet]


class HistoricalStats(BaseModel):
    """Mock historical risk/return statistics for a single ticker."""
    ticker: str
    avg_annual_return_pct: float
    volatility_pct: float
    max_drawdown_pct: float


class PositionInput(BaseModel):
    """Position specification used as input to the portfolio metrics tool."""
    ticker: str
    weight: float


class PortfolioMetrics(BaseModel):
    """Aggregate portfolio metrics returned by compute_portfolio_metrics."""
    expected_return_pct: float
    expected_volatility_pct: float
    comment: str


class StockExplanation(BaseModel):
    """
    Structured output for 'explain_stock' intent.
    The advisor will fill this using tool results.
    """
    intent: Literal["explain_stock"]
    ticker: str
    headline: str
    summary: str
    key_points: List[str]
    risk_factors: List[str]
    educational_disclaimer: str


class PortfolioPosition(BaseModel):
    """One position inside a suggested portfolio."""
    ticker: str
    weight: float
    rationale: str


class PortfolioSuggestion(BaseModel):
    """
    Structured output for 'suggest_portfolio' intent.
    The advisor will integrate tool results into these fields.
    """
    intent: Literal["suggest_portfolio"]
    positions: List[PortfolioPosition]
    overall_comment: str
    educational_disclaimer: str

class AgentState(BaseModel):
    # Conversation + routing
    conversation_history: List[str] = []
    intent: Optional[IntentType] = None
    stock_tickers: List[str] = []

    # User profile for portfolio suggestions
    profile: Profile = Profile()
    retrieved_chunks: List[str] = []
    final_answer: Optional[str] = None

    # NEW: keep structured outputs in state (useful for testing / later stages)
    stock_explanation: Optional[StockExplanation] = None
    portfolio_suggestion: Optional[PortfolioSuggestion] = None

    # Debug trace
    node_trace: List[str] = []
    
```
## Major required Functions and Nodes
```
@tool
def get_stock_profile(ticker: str) -> StockProfile:
    """
    Return a stock profile whose description is grounded in the vectorstore text.

    Flow:
    1. Normalize + validate ticker against the allowed universe.
    2. If we already built a profile for this ticker in this process, return it from cache.
    3. Query the Chroma vectorstore for relevant chunks about this ticker.
    4. Use the LLM in structured-output mode to turn those chunks into a StockProfile.
    5. If the vectorstore has no info, fall back to stocks_dict-based mock data.
    """

@tool
def get_recent_news(query: str) -> NewsResult:
    """
    Return 2–3 mock news snippets related to the query.
    This is fully mocked and does not actually hit the web.
    """

@tool
def get_historical_stats(ticker: str) -> HistoricalStats:
    """
    Return simple mock historical stats for the given ticker.
    The numbers are heuristic, based loosely on the risk level.
    Only tickers that exist in the vectorstore universe are allowed.
    """

@tool
def compute_portfolio_metrics(positions: List[PositionInput]) -> PortfolioMetrics:
    """
    Compute mock portfolio metrics from a list of (ticker, weight).
    We normalize weights and use a simple heuristic to produce expected return and volatility.
    Only tickers that exist in the vectorstore universe are allowed.
    """

def user_input_node(state: AgentState) -> AgentState:
    # Role: Gets the user's free-text investment request and appends it to conversation_history.
    # This is the only place where new user input enters the system in a given run.

def intent_classifier_node(state: AgentState) -> AgentState:
    # Role: Uses structured-output mode to classify the user's intent and extract tickers.

def collect_portfolio_specs_node(state: AgentState) -> AgentState:
    """
    Ensures that for 'suggest_portfolio' intent we always have:
    - profile.risk_level
    - profile.investment_horizon_years

    If they are missing, the node interactively asks the user for them and
    loops until both values are present. For non-portfolio intents it is a no-op.
    """

def tool_using_advisor_node(state: AgentState) -> AgentState:
    """
    Single advisor node that:
    - Sees the validated intent + user query + any RAG context (+ portfolio profile).
    - Can call multiple tools automatically via tool-calling.
    - Must finally return either StockExplanation or PortfolioSuggestion as JSON.
    - If a tool reports that a ticker is invalid, the advisor should choose a valid ticker
      from the allowed universe instead of failing.
    """

def final_node(state: AgentState) -> AgentState:
    # Role: Prints the final answer to the user and logs the full node_trace so you can
    # see which path the agent took through the graph for debugging and testing.

def update_profile_from_text(text: str, profile: Profile) -> Profile:
    """
    uses user input to update user profile (risk, horizon etc.)
    example user input: "horizon should be 5 years"
    Very simple heuristic extraction (no JSON / schema).
    """

```


## Test Scenarios
Perform the following test scenarious. Save all tests to a docx file (for each test, a record of user-agent interaction). Name of file: tests_[assignment_nr].docx
(e.g. tests_3.docx)
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



