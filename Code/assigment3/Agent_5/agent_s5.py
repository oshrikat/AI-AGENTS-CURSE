import os
import chromadb
from typing import Optional, List, Dict, Literal
from pydantic import BaseModel, Field
from google import genai
from dotenv import load_dotenv

# LangGraph runtime imports
from langgraph.graph import StateGraph, END

load_dotenv()

DB_PATH = r"C:\Users\oshri\Desktop\AI-AGENTS-CURSE\chroma_db"

# ==========================================
# 1. STRUCTURED OUTPUT SCHEMAS (Pydantic)
# ==========================================

class IntentResult(BaseModel):
    """
    מבנה הנתונים שג'מיני חייב להחזיר כשהוא מסווג את כוונת המשתמש.
    """
    intent: Literal["explain_stock", "suggest_portfolio", "unknown"]
    stock_tickers: List[str] = Field(description="List of stock tickers mentioned, if any")
    raw_question: str = Field(description="The original user question")


class Profile(BaseModel):
    """
    מבנה הנתונים של פרופיל ההשקעה.
    """
    risk_level: Optional[str] = None
    investment_horizon_years: Optional[str] = None
    budget_usd: Optional[str] = None


class RequirementsResult(BaseModel):
    """
    מבנה הנתונים שג'מיני חייב להחזיר כשהוא בודק מה חסר בפרופיל.
    """
    profile: Profile
    missing: List[Literal["risk_level", "investment_horizon_years", "budget_usd"]] = []

# ==========================================
# 2. AGENT STATE DEFINITION
# ==========================================

class AgentState(BaseModel):
    """
    הלוח המחיק המשותף של הסוכן (עודכן לשימוש באובייקטים המובנים).
    """
    messages: List[Dict[str, str]] = Field(default_factory=list)
    
    # שימוש בתוצאות המובנות במקום מחרוזות שבירות
    intent_result: Optional[IntentResult] = None
    requirements_result: Optional[RequirementsResult] = None
    
    retrieved_chunks: List[str] = Field(default_factory=list)
    final_answer: Optional[str] = None
    node_history: List[str] = Field(default_factory=list)

# ==========================================
# 3. INTERACTION & INTENT NODES
# ==========================================
from langchain_google_genai import ChatGoogleGenerativeAI

# אתחול המודל דרך LangChain כדי שנוכל להשתמש ביכולות הפלט המובנה
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def user_input_node(state: AgentState):
    print("\n--- [Node] User Input ---")
    user_message = input("You: ").strip()
    
    updated_messages = list(state.messages)
    updated_messages.append({"role": "user", "content": user_message})
    
    updated_history = list(state.node_history)
    updated_history.append("UserInput")
    
    return {"messages": updated_messages, "node_history": updated_history}

def intent_classifier_node(state: AgentState):
    print("\n--- [Node] Intent Classifier (Structured Output) ---")
    latest_user_message = state.messages[-1]["content"] if state.messages else ""
    
    # הפרומפט המעודכן - מגדיר גבולות גזרה ברורים מאוד למודל
    prompt = f"""
    Analyze the user's latest message and classify the intent into exactly ONE of the following categories:
    - 'explain_stock': The user asks for an explanation of specific stock(s) (e.g., "Tell me about NVDA").
    - 'suggest_portfolio': The user explicitly asks to build or suggest a portfolio.
    - 'unknown': The user asks a vague question, a general advice question (e.g., "Should I invest or just keep cash?"), or something unrelated.
    
    Extract any mentioned stock tickers.
    User message: "{latest_user_message}"
    """
    
    structured_llm = llm.with_structured_output(IntentResult)
    
    try:
        result: IntentResult = structured_llm.invoke(prompt)
    except Exception as e:
        print(f"[X] Intent classification failed: {e}")
        result = IntentResult(intent="unknown", stock_tickers=[], raw_question=latest_user_message)
        
    print(f"[DEBUG] Intent: {result.intent} | Tickers Extracted: {result.stock_tickers}")
    
    updated_history = list(state.node_history)
    updated_history.append("IntentClassifier")
    
    return {"intent_result": result, "node_history": updated_history}

# ==========================================
# 4. REQUIREMENTS & CLARIFICATION NODES
# ==========================================

def check_requirements_node(state: AgentState):
    print("\n--- [Node] Check Requirements (Structured Output) ---")
    
    # מעבירים ל-LLM את כל היסטוריית השיחה כדי שיבין מה נאמר עד כה
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state.messages])
    
    prompt = f"""
    Extract the investment profile from the conversation history.
    If a required field (risk_level, investment_horizon_years, budget_usd) is missing, add it to the 'missing' list.
    
    Conversation:
    {conversation_history}
    """
    
    # אילוץ המודל להחזיר אובייקט מסוג RequirementsResult
    structured_llm = llm.with_structured_output(RequirementsResult)
    
    try:
        result: RequirementsResult = structured_llm.invoke(prompt)
    except Exception as e:
         print(f"[X] Requirements extraction failed: {e}")
         result = RequirementsResult(profile=Profile(), missing=["risk_level", "investment_horizon_years", "budget_usd"])
         
    print(f"[DEBUG] Missing fields detected by LLM: {result.missing}")
    
    updated_history = list(state.node_history)
    updated_history.append("CheckRequirements")
    
    return {"requirements_result": result, "node_history": updated_history}

def ask_clarification_node(state: AgentState):
    print("\n--- [Node] Ask Clarification ---")
    # שולפים את רשימת החוסרים ישירות מהאובייקט המובנה
    missing_fields = state.requirements_result.missing if state.requirements_result else []
    
    print(f"Agent: To build a proper portfolio, I need a bit more info. Please provide: {', '.join(missing_fields)}")
    clarification = input("You (Clarification): ").strip()
    
    updated_messages = list(state.messages)
    updated_messages.append({"role": "user", "content": clarification})
    
    updated_history = list(state.node_history)
    updated_history.append("AskClarification")
    
    return {"messages": updated_messages, "node_history": updated_history}

def ask_rephrase_node(state: AgentState):
    print("\n--- [Node] Ask Rephrase ---")
    print("Agent: I can either explain specific stocks or build a portfolio. Could you please rephrase?")
    rephrase = input("You (Rephrase): ").strip()
    
    updated_messages = list(state.messages)
    updated_messages.append({"role": "user", "content": rephrase})
    
    updated_history = list(state.node_history)
    updated_history.append("AskRephrase")
    
    return {"messages": updated_messages, "node_history": updated_history}

# ==========================================
# 5. RAG EXECUTION NODES
# ==========================================
def rag_stock_node(state: AgentState):
    print("\n--- [Node] RAG Stock ---")
    # שולפים את סימולי המניות שחולצו על ידי הפלט המובנה!
    tickers = state.intent_result.stock_tickers if state.intent_result else []
    query = " ".join(tickers) if tickers else (state.intent_result.raw_question if state.intent_result else "")
    
    client = genai.Client()
    try:
        response = client.models.embed_content(model="gemini-embedding-2", contents=query)
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        collection = chroma_client.get_collection(name="stock_profiles")
        search_results = collection.query(query_embeddings=[response.embeddings[0].values], n_results=3)
        docs = search_results.get('documents', [[]])[0]
    except Exception as e:
        print(f"[X] RAG Error: {e}")
        docs = []

    updated_history = list(state.node_history)
    updated_history.append("RAG_Stock")
    return {"retrieved_chunks": docs, "node_history": updated_history}

def rag_portfolio_node(state: AgentState):
    print("\n--- [Node] RAG Portfolio ---")
    profile = state.requirements_result.profile
    query = f"Stocks suitable for {profile.risk_level} risk, {profile.investment_horizon_years} horizon, budget: {profile.budget_usd}"
    
    client = genai.Client()
    try:
        response = client.models.embed_content(model="gemini-embedding-2", contents=query)
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        collection = chroma_client.get_collection(name="stock_profiles")
        search_results = collection.query(query_embeddings=[response.embeddings[0].values], n_results=5)
        docs = search_results.get('documents', [[]])[0]
    except Exception as e:
        print(f"[X] RAG Error: {e}")
        docs = []

    updated_history = list(state.node_history)
    updated_history.append("RAG_Portfolio")
    return {"retrieved_chunks": docs, "node_history": updated_history}

# ==========================================
# 6. ANALYSIS & FINAL NODES
# ==========================================
def analysis_node(state: AgentState):
    print("\n--- [Node] Analysis ---")
    context_payload = "\n===\n".join(state.retrieved_chunks)
    intent = state.intent_result.intent if state.intent_result else "unknown"
    
    if intent == "suggest_portfolio":
        profile = state.requirements_result.profile
        system_instruction = f"Suggest a 3-position portfolio aligned with: Risk={profile.risk_level}, Horizon={profile.investment_horizon_years}, Budget={profile.budget_usd}."
    else:
        raw_q = state.intent_result.raw_question if state.intent_result else ""
        system_instruction = f"The user asked: '{raw_q}'. Explain ONLY the specific stock(s) requested. Ignore other stocks in the context."
        
    prompt = f"""
    You are a professional educational investment assistant.
    {system_instruction}
    
    Use ONLY the following retrieved context blocks to answer.
    Retrieved Context:
    {context_payload}
    
    Requirements:
    - Write a short, friendly intro.
    - Provide a structured explanation (strengths, risks, outlook) for the relevant stocks.
    - CRITICAL: Add a brief educational disclaimer at the very end stating this is not financial advice.
    - Be objective and clear.
    """
    
    try:
        # כאן אנחנו משתמשים במודל הרגיל של LangChain ליצירת טקסט חופשי וסיכום
        response = llm.invoke(prompt)
        answer = response.content.strip()
    except Exception as e:
        answer = f"Error: {e}"

    updated_history = list(state.node_history)
    updated_history.append("Analysis")
    return {"final_answer": answer, "node_history": updated_history}

def final_node(state: AgentState):
    updated_history = list(state.node_history)
    updated_history.append("Final")
    return {"node_history": updated_history}

# ==========================================
# 7. ROUTING FUNCTIONS (Using Structured Data!)
# ==========================================
def route_from_intent(state: AgentState) -> str:
    intent = state.intent_result.intent if state.intent_result else "unknown"
    if intent == "explain_stock":
        return "RAG_Stock"
    elif intent == "suggest_portfolio":
        return "CheckRequirements"
    else:
        return "AskRephrase"

def route_from_requirements(state: AgentState) -> str:
    # הלוגיקה פשוטה, הגיונית ולא דורשת שינון של מחרוזות. האם הרשימה ריקה? 
    if state.requirements_result and not state.requirements_result.missing:
        return "RAG_Portfolio"
    return "AskClarification"

# ==========================================
# 8. GRAPH ASSEMBLY
# ==========================================
def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    
    builder.add_node("UserInput", user_input_node)
    builder.add_node("IntentClassifier", intent_classifier_node)
    builder.add_node("CheckRequirements", check_requirements_node)
    builder.add_node("AskClarification", ask_clarification_node)
    builder.add_node("AskRephrase", ask_rephrase_node)
    builder.add_node("RAG_Stock", rag_stock_node)
    builder.add_node("RAG_Portfolio", rag_portfolio_node)
    builder.add_node("Analysis", analysis_node)
    builder.add_node("Final", final_node)
    
    builder.set_entry_point("UserInput")
    
    builder.add_edge("UserInput", "IntentClassifier")
    
    # חיווט עם פונקציית הניתוב החדשה
    builder.add_conditional_edges(
        "IntentClassifier", 
        route_from_intent,
        {
            "RAG_Stock": "RAG_Stock",
            "CheckRequirements": "CheckRequirements",
            "AskRephrase": "AskRephrase"
        }
    )
    
    builder.add_edge("AskRephrase", "IntentClassifier")
    
    # חיווט לולאת ההבהרה המבוססת על רשימת החוסרים המובנית
    builder.add_conditional_edges(
        "CheckRequirements", 
        route_from_requirements,
        {
            "AskClarification": "AskClarification",
            "RAG_Portfolio": "RAG_Portfolio"
        }
    )
    
    builder.add_edge("AskClarification", "CheckRequirements")
    
    builder.add_edge("RAG_Stock", "Analysis")
    builder.add_edge("RAG_Portfolio", "Analysis")
    builder.add_edge("Analysis", "Final")
    
    graph = builder.compile()
    
    try:
        os.makedirs("agents_plots", exist_ok=True)
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("agents_plots/agent_s5.png", "wb") as f:
            f.write(png_bytes)
    except Exception:
        pass
        
    return graph

def main():
    print("=== Starting Structured Output Agent (Stage 5) ===")
    graph = build_graph()
    state = AgentState()
    
    final_state = graph.invoke(state)
    
    print("\n==============================================")
    print("               FINAL REPORT                   ")
    print("==============================================")
    if final_state.get("final_answer"):
        print(final_state["final_answer"])
    else:
        print("No answer generated.")
        
    print("\n--- Agent Node Execution Trace (Debug) ---")
    print(" -> ".join(final_state.get("node_history", [])))
    print("==============================================")

if __name__ == "__main__":
    main()