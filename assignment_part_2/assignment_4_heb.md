# מטלה — (agent_s4.py): ניתוב מותנה (Conditional Routing)

## סקירה קומפקטית

גרסה זו של הסוכן (ראו `agents_plots/agent_s4.png`), שאותה עליכם ליצור, מממשת עוזר השקעות אינטראקטיבי בתור מכונת מצבים של LangGraph, המשתמשת ב־**conditional routing** כדי להחליט מה לעשות בהמשך לפי הודעת המשתמש ולפי המצב הנוכחי.

תחילה הסוכן מסווג את כוונת המשתמש: הסבר על מניה/מניות מסוימות, הצעת תיק השקעות, או בקשה לא ברורה. לאחר מכן הוא מנתב למסלולי צמתים שונים בהתאם לכך.

במסלול של הצעת תיק השקעות, הסוכן חוזר בלולאה מותנית כדי לשאול שאלות המשך עד שכל שדות הפרופיל הנדרשים — risk, horizon, budget — סופקו. לאחר מכן הוא ממשיך לשלב הבא.

בשני המסלולים המרכזיים, הסוכן שולף הקשר רלוונטי מתוך vector store ואז מייצר את התשובה הסופית. בנוסף, הוא מדפיס debug trace של הצמתים שבוקרו, כדי שתוכלו לראות את המסלול המדויק שנבחר.

## זרימת האינטראקציה

כדי להבין את זרימת המידע הרצויה, קראו את הסעיפים `Usage of Core Technology` ו־`Conditional Edges Explanation` מתוך `[agent_s4.md]`.

## המשימה

ממשו את הסוכן, תוך הקפדה שהוא כולל את כל הצמתים שמופיעים בדיאגרמה.

## מחלקות מסוימות שהסוכן צריך להשתמש בהן

```python
from pydantic import BaseModel
class Profile(BaseModel):
    risk_level: Optional[RiskLevel] = None
    investment_horizon_years: Optional[int] = None
    budget_usd: Optional[float] = None
```

## הערות הסבר על חלק מהצמתים

```python
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
