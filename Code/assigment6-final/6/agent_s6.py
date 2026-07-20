import os
import chromadb
import re
import json
from typing import Optional, List, Literal, Dict, Any
from pydantic import BaseModel, Field
from google import genai
from dotenv import load_dotenv

# LangChain & LangGraph imports for Tool Calling
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END


load_dotenv()

# נתיב יחסי למסד הנתונים - יוצא מהתיקייה הנוכחית החוצה עד לתיקיית השורש
DB_PATH = r"C:\Users\oshri\Desktop\AI-AGENTS-CURSE\chroma_db"

# ==========================================
# 1. STRUCTURED SCHEMAS (Pydantic Models)
# ==========================================
RiskLevel = Literal["low", "medium", "high"]
IntentType = Literal["explain_stock", "suggest_portfolio", "unknown"]

class Profile(BaseModel):
    risk_level: Optional[RiskLevel] = None
    investment_horizon_years: Optional[int] = None
    budget_usd: Optional[float] = None

class IntentResult(BaseModel):
    intent: IntentType
    stock_tickers: List[str]
    raw_question: str

class StockProfile(BaseModel):
    ticker: str
    name: str
    sector: str
    description: str
    risk_level: RiskLevel

class NewsSnippet(BaseModel):
    title: str
    snippet: str

class NewsResult(BaseModel):
    items: List[NewsSnippet]

class HistoricalStats(BaseModel):
    ticker: str
    avg_annual_return_pct: float
    volatility_pct: float
    max_drawdown_pct: float

class PositionInput(BaseModel):
    ticker: str
    weight: float

class PortfolioMetrics(BaseModel):
    expected_return_pct: float
    expected_volatility_pct: float
    comment: str

class StockExplanation(BaseModel):
    intent: str = Field(default="explain_stock", description="Must be exactly 'explain_stock'")
    ticker: str
    headline: str
    summary: str
    key_points: List[str]
    risk_factors: List[str]
    educational_disclaimer: str

class PortfolioPosition(BaseModel):
    ticker: str
    weight: float
    rationale: str

    
class PortfolioSuggestion(BaseModel):
    intent: str = Field(default="suggest_portfolio", description="Must be exactly 'suggest_portfolio'")
    positions: List[PortfolioPosition]
    overall_comment: str
    educational_disclaimer: str


class AgentState(BaseModel):
    conversation_history: List[str] = []
    intent: Optional[IntentType] = None
    stock_tickers: List[str] = []
    profile: Profile = Field(default_factory=Profile)
    retrieved_chunks: List[str] = []
    final_answer: Optional[str] = None
    
    # השדות המובנים לאחסון הפלט הסופי
    stock_explanation: Optional[StockExplanation] = None
    portfolio_suggestion: Optional[PortfolioSuggestion] = None
    
    node_trace: List[str] = []

# ==========================================
# 2. AGENT TOOLS (ארגז הכלים של הסוכן)
# ==========================================

@tool
def get_stock_profile(ticker: str) -> StockProfile:
    """
    מחזיר פרופיל מניה שהתיאור שלו מבוסס על הטקסט שב־ Vector Store.
    יש לקרוא לכלי זה תמיד לפני שממליצים על מניה.
    """
    print(f"[Tool Execution] Fetching profile from Vector Store for {ticker}...")
    client = genai.Client()
    try:
        response = client.models.embed_content(model="gemini-embedding-2", contents=ticker)
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        collection = chroma_client.get_collection(name="stock_profiles")
        search_results = collection.query(query_embeddings=[response.embeddings[0].values], n_results=2)
        docs = search_results.get('documents', [[]])[0]
        desc = " ".join(docs) if docs else "No detailed description found in database."
    except Exception as e:
        desc = f"Error retrieving data: {e}"

    # החזרת אובייקט נתונים מובנה כפי שהוגדר
    return StockProfile(
        ticker=ticker.upper(),
        name=f"{ticker.upper()} Corporation",
        sector="Technology/General", # ערך כללי לצורך ההדגמה
        description=desc,
        risk_level="medium"
    )

@tool
def get_recent_news(query: str) -> NewsResult:
    """מחזיר 2–3 תקצירי חדשות (Mock) הקשורים לשאילתה."""
    print(f"[Tool Execution] Fetching mock news for {query}...")
    return NewsResult(items=[
        NewsSnippet(title=f"Market Update: {query} Surges", snippet="Analysts note strong fundamentals and increasing market share."),
        NewsSnippet(title=f"{query} Faces New Regulations", snippet="Upcoming policy changes might introduce short-term volatility.")
    ])

@tool
def get_historical_stats(ticker: str) -> HistoricalStats:
    """מחזיר נתוני סיכון/תשואה היסטוריים (Mock) עבור המניה."""
    print(f"[Tool Execution] Calculating mock historical stats for {ticker}...")
    return HistoricalStats(
        ticker=ticker.upper(),
        avg_annual_return_pct=15.5,
        volatility_pct=22.0,
        max_drawdown_pct=-18.5
    )

@tool
def compute_portfolio_metrics(positions: List[PositionInput]) -> PortfolioMetrics:
    """מחשב מדדים מדומים של תיק השקעות מתוך רשימת פוזיציות."""
    print(f"[Tool Execution] Computing metrics for portfolio with {len(positions)} positions...")
    return PortfolioMetrics(
        expected_return_pct=12.0,
        expected_volatility_pct=18.5,
        comment="The suggested allocation provides a balanced risk/reward ratio based on historical mock data."
    )



# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def update_profile_from_text(text: str, profile: Profile) -> Profile:
    text_lower = text.lower()
    
    # בדיקת סיכון חכמה (מתעלמת משלילות כמו no high)
    if any(neg in text_lower for neg in ["no high", "not high", "no risk"]):
        profile.risk_level = "low"
    elif "low" in text_lower:
        profile.risk_level = "low"
    elif "medium" in text_lower:
        profile.risk_level = "medium"
    elif "high" in text_lower:
        profile.risk_level = "high"
        
    # חיפוש מספר שנים
    match = re.search(r'(\d+)\s*(year|yr)', text_lower)
    if match:
        profile.investment_horizon_years = int(match.group(1))
    elif "long term" in text_lower:
        profile.investment_horizon_years = 5
        
    return profile

# ==========================================
# 4. AGENT NODES
# ==========================================

def user_input_node(state: AgentState):
    print("\n--- [Node] User Input ---")
    user_message = input("You: ").strip()
    
    # חילוץ נתונים מיידי מכל הודעה
    updated_profile = update_profile_from_text(user_message, state.profile)
    
    updated_history = list(state.conversation_history)
    updated_history.append(user_message)
    
    updated_trace = list(state.node_trace)
    updated_trace.append("UserInput")
    
    return {
        "conversation_history": updated_history, 
        "node_trace": updated_trace,
        "profile": updated_profile  # שומרים את הנתונים ב-State באופן קבוע!
    }

def intent_classifier_node(state: AgentState):
    print("\n--- [Node] Intent Classifier ---")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    structured_llm = llm.with_structured_output(IntentResult)
    
    latest_msg = state.conversation_history[-1] if state.conversation_history else ""
    prompt = f"Analyze the user's request: '{latest_msg}'. Classify intent and extract any stock tickers."
    
    try:
        result = structured_llm.invoke(prompt)
    except Exception:
        result = IntentResult(intent="unknown", stock_tickers=[], raw_question=latest_msg)
        
    print(f"[DEBUG] Intent: {result.intent} | Tickers: {result.stock_tickers}")
    
    updated_trace = list(state.node_trace)
    updated_trace.append("IntentClassifier")
    return {"intent": result.intent, "stock_tickers": result.stock_tickers, "node_trace": updated_trace}

def collect_portfolio_specs_node(state: AgentState):
    print("\n--- [Node] Collect Portfolio Specs ---")
    updated_trace = list(state.node_trace)
    updated_trace.append("CollectPortfolioSpecs")
    
    # אם זה לא בקשת תיק, מדלגים
    if state.intent != "suggest_portfolio":
        return {"node_trace": updated_trace}
        
    profile = state.profile
    # לולאה אינטראקטיבית לאיסוף הנתונים החסרים
    while not profile.risk_level or not profile.investment_horizon_years:
        missing = []
        if not profile.risk_level: missing.append("risk level")
        if not profile.investment_horizon_years: missing.append("investment horizon (years)")
        
        print(f"Agent: To build a portfolio, I need more info. Please provide: {', '.join(missing)}")
        user_ans = input("You (Clarification): ").strip()
        profile = update_profile_from_text(user_ans, profile)
        
    return {"profile": profile, "node_trace": updated_trace}


def tool_using_advisor_node(state: AgentState):
    print("\n--- [Node] Tool Using Advisor (Strict Structured Mode) ---")
    
    tools = [get_stock_profile, get_recent_news, get_historical_stats, compute_portfolio_metrics]
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    # הגדרת הסכימה הנדרשת
    response_schema = StockExplanation if state.intent == "explain_stock" else PortfolioSuggestion
    
    # אכיפת סכימה נוקשה
    llm_with_tools = llm.bind_tools(tools).with_structured_output(response_schema)
    
    # System Prompt משודרג: עם הוראות תרגום מפורשות למודל
    system_text = f"""You are an expert investment advisor agent. 
    1. Always use tools to gather data. 
    2. Map tool outputs strictly to the required JSON fields.
    3. If the intent is explain_stock, map the tool's 'description' or 'outlook' fields to 'summary', 'key_points', and 'risk_factors' in the {response_schema.__name__} schema.
    4. NEVER output raw tool data structure if it differs from the required output schema.
    """
    
    messages = [SystemMessage(content=system_text)]
    for msg in state.conversation_history:
        messages.append(HumanMessage(content=msg))
    
    # לולאת הרצה
    ai_msg = llm_with_tools.invoke(messages)
    
    updated_trace = list(state.node_trace)
    updated_trace.append("ToolUsingAdvisor")
    
    if state.intent == "explain_stock":
        return {"stock_explanation": ai_msg, "final_answer": ai_msg.model_dump_json(), "node_trace": updated_trace}
    else:
        return {"portfolio_suggestion": ai_msg, "final_answer": ai_msg.model_dump_json(), "node_trace": updated_trace}

def final_node(state: AgentState):
    updated_trace = list(state.node_trace)
    updated_trace.append("Final")
    return {"node_trace": updated_trace}


# ==========================================
# 5. GRAPH ASSEMBLY & RUNTIME
# ==========================================

def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    
    # רישום כל הצמתים שבנינו
    builder.add_node("UserInput", user_input_node)
    builder.add_node("IntentClassifier", intent_classifier_node)
    builder.add_node("CollectPortfolioSpecs", collect_portfolio_specs_node)
    builder.add_node("ToolUsingAdvisor", tool_using_advisor_node)
    builder.add_node("Final", final_node)
    
    # חיווט הזרימה
    builder.set_entry_point("UserInput")
    builder.add_edge("UserInput", "IntentClassifier")
    
    # ניתוב חכם לפי כוונה
    def route_after_intent(state: AgentState):
        if state.intent == "suggest_portfolio":
            return "CollectPortfolioSpecs"
        return "ToolUsingAdvisor" # גם ל-explain_stock וגם ל-unknown (שילך ל-Advisor שיגיד שאין לו מה לעשות)
        
    builder.add_conditional_edges("IntentClassifier", route_after_intent)
    
    # חיבורים להמשך
    builder.add_edge("CollectPortfolioSpecs", "ToolUsingAdvisor")
    builder.add_edge("ToolUsingAdvisor", "Final")
    
    graph = builder.compile()
    
    # יצירת הדיאגרמה
    try:
        os.makedirs("agents_plots", exist_ok=True)
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("agents_plots/agent_s6.png", "wb") as f:
            f.write(png_bytes)
    except Exception:
        pass
        
    return graph

def main():
    print("=== Starting Agentic Tool-Use Investment Assistant (Stage 6) ===")
    graph = build_graph()
    state = AgentState()
    
    # הרצה
    final_state = graph.invoke(state)
    
    print("\n--- FINAL JSON OUTPUT ---")
    print(final_state.get("final_answer", "No answer generated."))

if __name__ == "__main__":
    main()