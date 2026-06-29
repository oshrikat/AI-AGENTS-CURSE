import os
import json
from typing import Optional, List
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

# LangGraph imports for building the state-driven agent pipeline
from langgraph.graph import StateGraph, END
from langgraph.types import Command

# Load investment API credentials from environment
load_dotenv()

# ==========================================
# 1. AGENT STATE DEFINITION (Shared Memory)
# ==========================================

class AgentState(BaseModel):
    """
    [ABOUT]
    The central Pydantic data schema representing the agent's short-term memory.
    It passes sequentially through all nodes, preserving state and intermediate data.
    """
    domain: Optional[str] = None
    risk: Optional[str] = None
    horizon: Optional[str] = None
    selected_stocks: List[str] = []
    final_answer: Optional[str] = None

# ==========================================
# 2. DETERMINISTIC CORE FUNCTIONS (Utils)
# ==========================================

def load_stock_profiles() -> dict:
    """
    [ABOUT]
    Loads raw stock information profiles from the provided dataset file.

    [FLOW]
    Reads JSON file -> Parses to Python dictionary -> Returns content.

    [INPUTS]
    - None (Reads straight from the workspace folder).

    [OUTPUTS]
    - dict: Full dictionary mapping stock tickers to their structural financial criteria.
    """
    filename = r"assignment_part_1\required_files\stocks_profiles.json"
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"[X] Error: The file {filename} was not found!")
        return {}


def filter_stocks_by_preferences(profiles: dict, domain: str, risk_level: str) -> List[str]:
    """
    [ABOUT]
    Rule-based filtering logic that evaluates candidates purely on fixed conditions.

    [FLOW]
    Iterates through profiles -> Matches domain and risk -> Appends successful Tickers to list.

    [INPUTS]
    - profiles (dict): Complete loaded stock universe dictionary.
    - domain (str): The target sector provided by the user.
    - risk_level (str): The desired risk parameter (low, medium, high).

    [OUTPUTS]
    - List[str]: A list of ticker symbols (e.g., ['XOM', 'CVX']) matching the criteria.
    """
    selected_tickers = []
    target_domain = domain.strip().lower()
    target_risk = risk_level.strip().lower()
    
    for ticker, info in profiles.items():
        stock_domain = info.get("domain", "").strip().lower()
        stock_risk = info.get("risk_level", "").strip().lower()
        
        if stock_domain == target_domain and stock_risk == target_risk:
            selected_tickers.append(ticker)
            
    return selected_tickers

# ==========================================
# 3. LANGGRAPH AGENT NODES
# ==========================================

def preference_collector_node(state: AgentState) -> Command:
    """
    [ABOUT]
    Node #1: Responsible for interacting with the user and initializing state values.

    [FLOW]
    Receives current state -> Prompts CLI inputs -> Encapsulates inputs in a Command -> Routes to StockSelector.

    [INPUTS]
    - state (AgentState): The active state object at execution start.

    [OUTPUTS]
    - Command: Updates 'domain', 'risk', and 'horizon' keys and triggers the transition to 'StockSelector'.
    """
    print("\n--- [Node] Preference Collector ---")
    domain = input("Enter preferred investment domain (e.g., Energy, Tech): ").strip()
    risk = input("Enter desired risk level (low, medium, high): ").strip().lower()
    horizon = input("Enter investment horizon (Short-term, Long-term): ").strip()
    
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
    Node #2: Executes deterministic asset filtering within the agent architecture.

    [FLOW]
    Extracts criteria from state -> Runs rule-based utility function -> Updates state list -> Routes to LLMExplanation.

    [INPUTS]
    - state (AgentState): Contains user-defined preference parameters.

    [OUTPUTS]
    - Command: Updates 'selected_stocks' field with matched tickers and advances workflow to 'LLMExplanation'.
    """
    print("\n--- [Node] Stock Selector ---")
    profiles = load_stock_profiles()
    
    matching_tickers = filter_stocks_by_preferences(
        profiles=profiles, 
        domain=state.domain, 
        risk_level=state.risk
    )
    
    print(f"[V] Found matching stocks: {matching_tickers}")
    
    return Command(
        update={"selected_stocks": matching_tickers},
        goto="LLMExplanation"
    )


def llm_explanation_node(state: AgentState) -> Command:
    """
    [ABOUT]
    Node #3: Generates natural language insights utilizing the Gemini API engine.

    [FLOW]
    Checks for empty matches (triggers fallback apology if empty) -> Compiles structured prompt context 
    -> Dispatches request to Gemini -> Commits text payload to state -> Shuts down workflow by routing to END.

    [INPUTS]
    - state (AgentState): Holds aggregated profile matches and target profile settings.

    [OUTPUTS]
    - Command: Commits the final generated presentation string to 'final_answer' and exits the graph pipeline.
    """
    print("\n--- [Node] LLM Explanation ---")
    
    # Trigger absolute apology fallback clause if zero assets matched filtering criteria
    if not state.selected_stocks:
        apology_message = "Sorry, none of the provided stocks matches your criteria."
        return Command(
            update={"final_answer": apology_message},
            goto=END
        )
        
    client = genai.Client()
    profiles = load_stock_profiles()
    
    stocks_context = ""
    for ticker in state.selected_stocks:
        info = profiles.get(ticker, {})
        stocks_context += f"\n- Stock: {ticker}\n  Outlook: {info.get('outlook')}\n  Risks: {', '.join(info.get('key_risks', []))}\n"

    prompt = f"""
    You are a professional educational investment assistant.
    Generate a formatted portfolio suggestion report based on the following details.
    
    User Criteria:
    - Domain: {state.domain}
    - Risk Level: {state.risk}
    - Horizon: {state.horizon}
    
    Selected Stocks Data:
    {stocks_context}
    
    Requirements:
    1. Write a short intro acknowledging the user's preferences.
    2. Create a bullet-point list for each selected stock ticker.
    3. For each bullet, provide 2-3 sentences explaining why it fits their horizon and risk profile.
    4. Maintain an educational, objective tone. Do not give direct buy/sell financial advice.
    5. Answer in plain English.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        answer = response.text.strip()
    except Exception as e:
        answer = f"Error generating report from LLM: {e}"
        
    return Command(
        update={"final_answer": answer},
        goto=END
    )

# ==========================================
# 4. GRAPH ASSEMBLY & RUNTIME EXECUTION
# ==========================================

def build_graph() -> StateGraph:
    """
    [ABOUT]
    Orchestrates the construction, functional linking, and compilation of the LangGraph agent.

    [FLOW]
    Instantiates StateGraph -> Registers 3 functional nodes -> Sets starting node -> Saves visualization image -> Returns runtime graph object.

    [INPUTS]
    - None.

    [OUTPUTS]
    - CompiledGraph: An executable application graph instance.
    """
    builder = StateGraph(AgentState)
    
    # Register functional node modules inside graph logic mapping
    builder.add_node("PreferenceCollector", preference_collector_node)
    builder.add_node("StockSelector", stock_selector_node)
    builder.add_node("LLMExplanation", llm_explanation_node)
    
    # Establish entry route logic
    builder.set_entry_point("PreferenceCollector")
    
    graph = builder.compile()
    
    # Automatically export structural visualization schematic blueprint to workspace disk
    try:
        os.makedirs("agents_plots", exist_ok=True)
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("agents_plots/agent_s2.png", "wb") as f:
            f.write(png_bytes)
        print("[V] Success: Agent diagram saved to 'agents_plots/agent_s2.png'")
    except Exception as e:
        print(f"[-] Note: Could not generate graph image: {e}")
        
    return graph


def main():
    """
    [ABOUT]
    Main CLI orchestrator running the linear agent framework lifecycle.
    """
    print("=== Starting LangGraph Investment Agent ===")
    graph = build_graph()
    state = AgentState()
    
    # Run the compiled graph end-to-end
    final_state = graph.invoke(state)
    
    # Output presentation results to the terminal window
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