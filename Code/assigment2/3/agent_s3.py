import os
import chromadb
from typing import Optional, List
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

# LangGraph imports for agent structure
from langgraph.graph import StateGraph, END
from langgraph.types import Command

# Load credentials
load_dotenv()

# ==========================================
# 1. AGENT STATE DEFINITION (With RAG Context)
# ==========================================

class AgentState(BaseModel):
    """
    [ABOUT]
    The updated data schema representing the agent's short-term memory.
    
    [CHANGES FOR STAGE 3]
    Added 'retrieved_chunks' to hold the text blocks retrieved from Chroma 
    so the downstream LLM node can utilize them as grounding context.
    """
    domain: Optional[str] = None
    risk: Optional[str] = None
    horizon: Optional[str] = None
    selected_stocks: List[str] = []       # Will hold tickers extracted/found during process
    retrieved_chunks: List[str] = []      # NEW: Grounding blocks from semantic search
    final_answer: Optional[str] = None

# ==========================================
# 2. LANGGRAPH AGENT NODES
# ==========================================

def preference_collector_node(state: AgentState) -> Command:
    """
    [ABOUT]
    Node #1: Collects target criteria from the user via interactive CLI.

    [FLOW]
    Receives empty state -> Prompts CLI inputs -> Encapsulates inputs in Command -> Routes to StockSelector.

    [INPUTS]
    - state (AgentState): Active execution state.

    [OUTPUTS]
    - Command: Updates preferences and triggers transition to 'StockSelector'.
    """
    print("\n--- [Node] Preference Collector ---")
    
    # 1. Validation loop for Domain
    valid_domains = ["tech", "health", "finance", "energy", "consumer_goods"]
    while True:
        domain = input("Enter preferred investment domain (e.g., tech, health): ").strip().lower()
        if domain in valid_domains:
            break
        # הודעת השגיאה המדויקת שהמרצה דרש
        print("choose from tech, health, finance, energy, consumer_goods")

    # 2. Validation loop for Risk
    valid_risks = ["low", "medium", "high"]
    while True:
        risk = input("Enter desired risk level (low, medium, high): ").strip().lower()
        if risk in valid_risks:
            break
        # הודעת השגיאה המדויקת שהמרצה דרש
        print("Please type 'low', 'medium', or 'high'.")

    # 3. Horizon collection (no strict validation specified in the scenario, but good practice)
    horizon = input("Enter investment horizon (Short-term, Long-term): ").strip()
    
    print("\n[V] Preferences collected successfully.")
    
    return Command(
        update={
            "domain": domain,
            "risk": risk,
            "horizon": horizon
        },
        goto="StockSelector"
    )


def stock_selector_node(state: AgentState) -> Command:
    """
    [ABOUT]
    Node #2: Performs semantic retrieval (RAG) using the local Chroma Vector Store.

    [FLOW]
    Assembles user preferences query -> Generates query embedding via Gemini 
    -> Executes similarity search in Chroma DB -> Extracts top 4 relevant context blocks 
    -> Saves chunks and proceeds to LLMExplanation.

    [INPUTS]
    - state (AgentState): Holds user profile constraints.

    [OUTPUTS]
    - Command: Updates 'retrieved_chunks' and advances workflow to 'LLMExplanation'.
    """
    print("\n--- [Node] Stock Selector (Semantic RAG Mode) ---")
    
    # 1. בניית השאילתה הסמנטית מהעדפות המשתמש
    query_parts = []
    if state.domain: query_parts.append(f"domain: {state.domain}")
    if state.risk: query_parts.append(f"risk level: {state.risk}")
    if state.horizon: query_parts.append(f"horizon: {state.horizon}")
    query = ", ".join(query_parts) if query_parts else "general stock information"
    
    print(f"[DEBUG] Assembled Semantic Search Query: '{query}'")
    
    client = genai.Client()
    
    try:
        # 2. המרת השאילתה לווקטור באותו מודל איתו בנינו את המאגר
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=query
        )
        query_embedding = response.embeddings[0].values
        
        # 3. התחברות ל-Chroma המקומי ושליפת 4 הבלוקים הקרובים ביותר
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection(name="stock_profiles")
        
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=4
        )
        
        docs = search_results.get('documents', [[]])[0]
        metadatas = search_results.get('metadatas', [[]])[0]
        
        # הדפסת הבלוקים שנמצאו למטרת דיבאג (דרישה של המרצה לבדיקה)
        print("\n--- DEBUG: Retrieved Chunks From Chroma DB ---")
        discovered_tickers = []
        for i, (doc, meta) in enumerate(zip(docs, metadatas), 1):
            ticker = meta.get("ticker", "UNKNOWN")
            discovered_tickers.append(ticker)
            print(f" Chunk {i} [{ticker}]: {doc[:120].replace('\n', ' ')}...")
            
        return Command(
            update={
                "retrieved_chunks": docs,
                "selected_stocks": discovered_tickers
            },
            goto="LLMExplanation"
        )
        
    except Exception as e:
        print(f"[X] RAG Node failed: {e}")
        # פתרון מגננה למקרה של תקלה ברשת
        return Command(
            update={"final_answer": f"Retrieval phase error: {e}"},
            goto=END
        )


def llm_explanation_node(state: AgentState) -> Command:
    """
    [ABOUT]
    Node #3: Synthesizes the final grounded educational report using the retrieved knowledge blocks.

    [FLOW]
    Combines retrieved chunks into a context block -> Injects constraints into a generation prompt 
    -> Requests Gemini to draft the response -> Updates state final_answer -> Finishes.

    [INPUTS]
    - state (AgentState): Holds aggregated profile matches and retrieved text context.

    [OUTPUTS]
    - Command: Updates 'final_answer' and exits the graph pipeline.
    """
    print("\n--- [Node] LLM Explanation (Grounded Generation) ---")
    
    if not state.retrieved_chunks:
        return Command(
            update={"final_answer": "Error: No context was retrieved to back the generation."},
            goto=END
        )
        
    #חיבור כל הבלוקים שנמצאו ב-Chroma לבלוק מידע אחד המזין את ג'מיני
    context_payload = "\n===\n".join(state.retrieved_chunks)
    
    prompt = f"""
    You are a professional educational investment assistant.
    Generate a formatted portfolio suggestion report based ONLY on the provided retrieved stock chunks context.
    
    User Target Criteria:
    - Domain: {state.domain}
    - Risk Level: {state.risk}
    - Horizon: {state.horizon}
    
    Retrieved Stocks Grounding Context:
    {context_payload}
    
    Requirements:
    1. Write a friendly intro acknowledging the user's preferences.
    2. Suggest up to 3 concrete stocks that appear in the retrieved chunks and best match their goals.
    3. For each suggested stock, write a bullet point providing 2-3 sentences explaining why it fits based on the context.
    4. If the retrieved chunks are weakly related or do not contain suitable matches for the criteria, you MUST explicitly output: "Sorry, none of the retrieved information matches your criteria well enough." Do not invent or fabricate stock tickers.
    5. Maintain a strictly objective, educational tone. No professional financial advice. Answers must be in clear plain English.
    """
    
    client = genai.Client()
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        answer = response.text.strip()
    except Exception as e:
        answer = f"Generation phase error: {e}"
        
    return Command(
        update={"final_answer": answer},
        goto=END
    )

# ==========================================
# 3. GRAPH & RUNTIME EXECUTION
# ==========================================

def build_graph() -> StateGraph:
    """
    [ABOUT]
    Orchestrates the construction and functional linking of the LangGraph RAG machine.
    """
    builder = StateGraph(AgentState)
    
    # Register graph nodes
    builder.add_node("PreferenceCollector", preference_collector_node)
    builder.add_node("StockSelector", stock_selector_node)
    builder.add_node("LLMExplanation", llm_explanation_node)
    
    # Wire the pipeline logic
    builder.set_entry_point("PreferenceCollector")
    
    graph = builder.compile()
    
    # Save diagram as agent_s3.png
    try:
        os.makedirs("agents_plots", exist_ok=True)
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("agents_plots/agent_s3.png", "wb") as f:
            f.write(png_bytes)
        print("[V] Success: Agent diagram saved to 'agents_plots/agent_s3.png'")
    except Exception as e:
        print(f"[-] Note: Could not generate graph image: {e}")
        
    return graph


def main():
    print("=== Starting LangGraph RAG Investment Agent (Stage 3) ===")
    graph = build_graph()
    state = AgentState()
    
    # Run graph execution pipeline
    final_state = graph.invoke(state)
    
    print("\n==============================================")
    print("               FINAL REPORT                   ")
    print("==============================================")
    if final_state.get("final_answer"):
        print(final_state["final_answer"])
    else:
        print("No answer generated.")
    print("==============================================")


if __name__ == "__main__":
    main()