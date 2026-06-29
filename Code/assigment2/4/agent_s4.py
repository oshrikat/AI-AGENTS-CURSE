import os
import chromadb
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from google import genai
from dotenv import load_dotenv

# LangGraph runtime imports
from langgraph.graph import StateGraph, END
from langgraph.types import Command

load_dotenv()

# ==========================================
# 1. UNDERLYING DATA SCHEMAS (Pydantic)
# ==========================================

class UserProfile(BaseModel):
    """
    [ABOUT]
    Holds the required parameters needed to construct a financial portfolio.
    
    [FIELDS]
    - risk: Investment risk tolerance (low, medium, high).
    - horizon: Duration context (Short-term, Long-term).
    - budget: Financial scope string (e.g., "$10k", "5000").
    """
    risk: Optional[str] = None
    horizon: Optional[str] = None
    budget: Optional[str] = None


class AgentState(BaseModel):
    """
    [ABOUT]
    The advanced, nested memory schema for our multi-intent branching agent.
    
    [FLOW CONTEXT]
    Every node in the graph receives this object, can read the history, 
    update the profile, modify the current intent, and log its own footprint.
    """
    # היסטוריית השיחה: נשמור רשימה של מילונים המייצגים את הצ'אט
    # דוגמה: [{"role": "user", "content": "hello"}, {"role": "model", "content": "Hi!"}]
    messages: List[Dict[str, str]] = Field(default_factory=list)
    
    # כוונת המשתמש הנוכחית כפי שתסווג על ידי ה-LLM (explain_stock / suggest_portfolio / unknown)
    intent: Optional[str] = None
    
    # הפרופיל המצטבר (מכיל סיכון, אופק ותקציב)
    profile: UserProfile = Field(default_factory=UserProfile)
    
    # רשימת המניות שנמצאו/חולצו במהלך הריצה
    selected_stocks: List[str] = Field(default_factory=list)
    
    # הבלוקים הטקסטואליים שיישלפו מה-Chroma DB הקיים שלנו
    retrieved_chunks: List[str] = Field(default_factory=list)
    
    # התשובה הסופית שהסוכן יציג למשתמש על המסך
    final_answer: Optional[str] = None
    
    # דרישת המרצה: מעקב מפורש אחרי סדר התחנות (Nodes) שהסוכן ביקר בהן
    node_history: List[str] = Field(default_factory=list)

# ==========================================
# 2. CONSOLE INTERACTION & INPUT NODE
# ==========================================
def user_input_node(state: AgentState):
    print("\n--- [Node] User Input ---")
    user_message = input("You: ").strip()
    
    updated_messages = list(state.messages)
    updated_messages.append({"role": "user", "content": user_message})
    updated_history = list(state.node_history)
    updated_history.append("UserInput")
    
    return {"messages": updated_messages, "node_history": updated_history}

# ==========================================
# 3. INTENT CLASSIFICATION NODE
# ==========================================
def intent_classifier_node(state: AgentState):
    print("\n--- [Node] Intent Classifier ---")
    latest_user_message = state.messages[-1]["content"] if state.messages else ""
    
    prompt = f"""
    You are an intent classification routing tool for a financial assistant.
    Analyze the user's latest message and classify it into exactly ONE of the following categories:
    
    1. 'explain_stock' -> specific stock ticker (e.g., "Tell me about NVDA").
    2. 'suggest_portfolio' -> general suggestion (e.g., "Build me a portfolio").
    3. 'unknown' -> vague or unrelated (e.g., "hello", "what is a stack?").
    
    User message: "{latest_user_message}"
    Respond with ONLY the category string.
    """
    client = genai.Client()
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        detected_intent = response.text.strip().lower()
    except Exception:
        detected_intent = "unknown"
        
    print(f"[DEBUG] LLM classified user intent as: '{detected_intent}'")
    updated_history = list(state.node_history)
    updated_history.append("IntentClassifier")
    
    return {"intent": detected_intent, "node_history": updated_history}

# ==========================================
# 4. REQUIREMENTS CHECK & CLARIFICATION NODES
# ==========================================
def check_requirements_node(state: AgentState):
    print("\n--- [Node] Check Requirements ---")
    missing = []
    if not state.profile.risk: missing.append("risk")
    if not state.profile.horizon: missing.append("horizon")
    if not state.profile.budget: missing.append("budget")
    
    updated_history = list(state.node_history)
    updated_history.append("CheckRequirements")
    
    if missing:
        print(f"[DEBUG] Profile incomplete. Missing: {missing}")
    else:
        print("[V] Profile complete! Advancing.")
        
    return {"node_history": updated_history}

def ask_clarification_node(state: AgentState):
    print("\n--- [Node] Ask Clarification Loop ---")
    print("Agent: To build a portfolio, please specify risk, horizon, and budget.")
    clarification = input("You (Clarification): ").strip().lower()
    
    updated_profile = UserProfile(**state.profile.model_dump())
    if "low" in clarification: updated_profile.risk = "low"
    elif "medium" in clarification: updated_profile.risk = "medium"
    elif "high" in clarification: updated_profile.risk = "high"
    if "short" in clarification: updated_profile.horizon = "Short-term"
    elif "long" in clarification: updated_profile.horizon = "Long-term"
    if "$" in clarification or any(char.isdigit() for char in clarification):
        updated_profile.budget = clarification
        
    updated_history = list(state.node_history)
    updated_history.append("AskClarification")
    return {"profile": updated_profile, "node_history": updated_history}

def ask_rephrase_node(state: AgentState):
    print("\n--- [Node] Ask Rephrase Loop ---")
    print("Agent: I'm sorry, I didn't quite catch that. Could you please rephrase?")
    updated_history = list(state.node_history)
    updated_history.append("AskRephrase")
    return {"node_history": updated_history}

# ==========================================
# 5. RAG EXECUTION NODES
# ==========================================
def rag_stock_node(state: AgentState):
    print("\n--- [Node] RAG Stock ---")
    latest_user_message = state.messages[-1]["content"] if state.messages else ""
    client = genai.Client()
    try:
        response = client.models.embed_content(model="gemini-embedding-2", contents=latest_user_message)
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection(name="stock_profiles")
        search_results = collection.query(query_embeddings=[response.embeddings[0].values], n_results=3)
        docs = search_results.get('documents', [[]])[0]
    except Exception:
        docs = []

    updated_history = list(state.node_history)
    updated_history.append("RAG_Stock")
    return {"retrieved_chunks": docs, "node_history": updated_history}

def rag_portfolio_node(state: AgentState):
    print("\n--- [Node] RAG Portfolio ---")
    query = f"Stocks suitable for {state.profile.risk} risk, {state.profile.horizon} horizon, budget: {state.profile.budget}"
    client = genai.Client()
    try:
        response = client.models.embed_content(model="gemini-embedding-2", contents=query)
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection(name="stock_profiles")
        search_results = collection.query(query_embeddings=[response.embeddings[0].values], n_results=5)
        docs = search_results.get('documents', [[]])[0]
    except Exception:
        docs = []

    updated_history = list(state.node_history)
    updated_history.append("RAG_Portfolio")
    return {"retrieved_chunks": docs, "node_history": updated_history}

# ==========================================
# 6. LLM SYNTHESIS & FINAL NODES
# ==========================================
def analysis_node(state: AgentState):
    print("\n--- [Node] Analysis ---")
    context_payload = "\n===\n".join(state.retrieved_chunks)
    
    # שולפים את הודעת המשתמש האחרונה כדי שג'מיני ידע בדיוק מה נשאל
    latest_user_message = state.messages[-1]["content"] if state.messages else ""
    
    if state.intent == "suggest_portfolio":
        system_instruction = f"Suggest a 3-position portfolio aligned with: Risk={state.profile.risk}, Horizon={state.profile.horizon}, Budget={state.profile.budget}."
    else:
        system_instruction = f"The user asked: '{latest_user_message}'. Explain ONLY the specific stock(s) requested. Ignore other stocks in the context."
        
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
    
    client = genai.Client()
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        answer = response.text.strip()
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
# CONDITIONAL ROUTING FUNCTIONS
# ==========================================

def route_intent(state: AgentState) -> str:
    """
    [ABOUT]
    First router: decides the main branch based on LLM classification.
    """
    intent = state.intent
    if intent == "explain_stock":
        return "RAG_Stock"
    elif intent == "suggest_portfolio":
        return "CheckRequirements"
    else:
        return "AskRephrase"

def route_requirements(state: AgentState) -> str:
    """
    [ABOUT]
    Second router: creates the clarification loop. 
    """
    profile = state.profile
    if not profile.risk or not profile.horizon or not profile.budget:
        return "AskClarification"
    return "RAG_Portfolio"


# ==========================================
#  GRAPH ASSEMBLY & EXECUTION
# ==========================================

def build_graph() -> StateGraph:
    """
    [ABOUT]
    Wires the nodes, sets conditional edges, and compiles the agent.
    """
    builder = StateGraph(AgentState)
    
    # 1. רישום כל התחנות (הצמתים) בגרף
    builder.add_node("UserInput", user_input_node)
    builder.add_node("IntentClassifier", intent_classifier_node)
    builder.add_node("CheckRequirements", check_requirements_node)
    builder.add_node("AskClarification", ask_clarification_node)
    builder.add_node("AskRephrase", ask_rephrase_node)
    builder.add_node("RAG_Stock", rag_stock_node)
    builder.add_node("RAG_Portfolio", rag_portfolio_node)
    builder.add_node("Analysis", analysis_node)
    builder.add_node("Final", final_node)
    
    # 2. הגדרת נקודת ההתחלה
    builder.set_entry_point("UserInput")
    
# 3. חיבור הקשתות והלולאות (מעודכן ותקין)
    builder.add_edge("UserInput", "IntentClassifier")
    builder.add_conditional_edges("IntentClassifier", route_intent)
    
    # לולאת חזרה 1: אם המשתמש לא הובן, נחזיר אותו לתחילת המסלול כדי שיקליד שוב
    builder.add_edge("AskRephrase", "UserInput")
    
    builder.add_conditional_edges("CheckRequirements", route_requirements)
    
    # לולאת חזרה 2: אם חסרים נתונים, נחזור לבדוק שוב אחרי שהוא עונה
    builder.add_edge("AskClarification", "CheckRequirements")
    
    builder.add_edge("RAG_Stock", "Analysis")
    builder.add_edge("RAG_Portfolio", "Analysis")
    builder.add_edge("Analysis", "Final")
    
    graph = builder.compile()
    
    # הפקת תמונת דיאגרמה המציגה את הגרף המסועף החדש
    try:
        os.makedirs("agents_plots", exist_ok=True)
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("agents_plots/agent_s4.png", "wb") as f:
            f.write(png_bytes)
        print("[V] Success: Multi-Intent Agent diagram saved to 'agents_plots/agent_s4.png'")
    except Exception as e:
        print(f"[-] Note: Could not generate graph image: {e}")
        
    return graph


def main():
    print("=== Starting Interactive LangGraph Investment Agent (Stage 4) ===")
    graph = build_graph()
    
    # אתחול הזיכרון המשותף
    state = AgentState()
    
    # הרצת הסוכן (הוא יעצור ויבקש קלט בזכות ה-input() בתוך הצמתים)
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





