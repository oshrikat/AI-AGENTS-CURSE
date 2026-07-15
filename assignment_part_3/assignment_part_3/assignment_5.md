# Assignment — Stage 5: Structured Output (modification of stage-4 agent)

## Compact Overview

This investment consultant (see agents_plots/agent_s5.png) is modified version of agent_s4 that is supposed to illustrate **structured output** by using `llm.with_structured_output(...)`  (langgraph 1.0.4) to force the LLM to return typed Pydantic objects for both intent classification (`IntentResult`) and requirements checking (`RequirementsResult`). Those structured results are written into a shared `AgentState`, enabling reliable conditional routing (e.g., ask for clarifications only when required fields are missing) instead of brittle string parsing. 

## Interaction Flow
To understand the necessary modifications to stage-4 agent read the section "Usage of Core Technology" from [explanation_agent_s5.md]

## Task
Implement the necessary modifications.


## some classes that were not part of the previous agent
from pydantic import BaseModel
```
class IntentResult(BaseModel):
    intent: IntentType
    stock_tickers: List[str]
    raw_question: str


class RequirementsResult(BaseModel):
    profile: Profile
    missing: List[Literal["risk_level", "investment_horizon_years", "budget_usd"]] = []
```

## Test Scenarios
Perform the following test scenarious. Save all tests to a docx file (for each test, a record of user-agent interaction). Name of file: tests_[assignment_nr].docx
(e.g. tests_3.docx)
### Scenario 1 — Stock explanation (single-turn)

**User input**:

“What do you think about NVDA? Main risks and outlook please.”

**Expected agent response** (high-level):

Produces a clear explanation of strengths / risks / outlook for NVDA, grounded in retrieved context chunks.

Ends with a short educational disclaimer.

(It does not ask profile questions.)

Expected flow (implicit): IntentClassifier → RAG_Stock → Analysis → Final

### Scenario 2 — Portfolio request with missing fields (clarification loop)

**User input** (turn 1):

“Build me a portfolio of 3 stocks.”

**Expected agent response** (turn 1):

Asks for missing details needed to proceed, e.g.: risk level + horizon + budget (in one message).

**User input** (turn 2):

“Medium risk, 5 years, $10k.”

**Expected agent response** (turn 2, high-level):

Suggests a 3-position portfolio matching the profile.

Gives 2–3 sentences per stock explaining fit.

(No disclaimer is required in the portfolio branch in this version.)

Expected flow (implicit):
IntentClassifier → CheckRequirements → AskClarification → CheckRequirements → RAG_Portfolio → Analysis → Final

### Scenario 3 — Unknown intent → rephrase → stock explanation

**User input** (turn 1):

“Should I invest or just keep cash?”

**Expected agent response** (turn 1):

States it can either explain a specific stock or suggest a portfolio, and asks the user to rephrase accordingly.

**User input** (turn 2):

“Ok—explain TSLA for me.”

**Expected agent response** (turn 2, high-level):

Provides TSLA strengths/risks/outlook using retrieved chunks + ends with an educational disclaimer.

Expected flow (implicit):
IntentClassifier → AskRephrase → IntentClassifier → RAG_Stock → Analysis → Final
