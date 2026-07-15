import os
import json
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_stock_profiles() -> dict:
    """
    [הסבר על הפונקציה]
    Loads the financial profiles of all available stocks from a local JSON file[stocks_profiles.json].

    [איך הפונקציה פועלת]
    1. Defines the file path where the JSON data is stored.
    2. Opens and reads the file using UTF-8 encoding
    3. Parses the raw JSON content into a native Python dictionary
    4. Handles potential file-not-found errors safely.

    [קלטים]
    - None - not getting any inputs parameters

    [מה הפונקציה מחזירה]
    - dict: A dictionary containing all stock information where keys are Tickers (e.g., "XOM")
    """

    filename = r"required_files\stocks_profiles.json"
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            profiles = json.load(file)
            print(f"[V] Success: Loaded {len(profiles)} stock profiles.")
            return profiles
    except FileNotFoundError:
        print(f"[X] Error: The file {filename} was not found!")
        return {}

def ask_user_preferences() -> dict:
    """
    [הסבר על הפונקציה]
    Interacts with the user via the console to collect their investment criteria .

    [איך הפונקציה פועלת - flow ]
    1. Prompts the user to type their preferred industry sector (domain) 
    2. Prompts the user to type their acceptable risk level
    3. Prompts the user to type their preferred investment duration (horizon) 
    4. Cleans the textual inputs by removing unnecessary spaces and converting to lowercase.
    5. Packages the answers into a structured dictionary.

    [קלט]
    - None (Takes live user input from the keyboard).

    [פלט]
    - dict: A dictionary with keys: 'domain', 'risk_level', and 'horizon'.
    """
    print("\n--- Collect Investment Preferences ---")
    
    # Get user inputs via console
    domain = input("Enter preferred investment domain (e.g., Energy, Tech): ").strip()
    risk_level = input("Enter desired risk level (low, medium, high): ").strip().lower()
    horizon = input("Enter investment horizon (Short-term, Long-term): ").strip()
    
    # Return structured dictionary  
    preferences = {
        "domain": domain,
        "risk_level": risk_level,
        "horizon": horizon
    }
    
    return preferences

def filter_stocks(profiles: dict, prefs: dict) -> dict:
    """
    [הסבר על הפונקציה]
    Applies a deterministic rule-based filter to match stock data with user desires .

    [איך הפונקציה פועלת - flow ]
    1. Extracts the target domain and risk level from the user preferences dictionary 
    2. Loops through every single stock available in the loaded profiles 
    3. Checks if a stock's industry AND risk level exactly match what the user wants.
    4. Saves matching stocks into a new temporary collection .
    5. Returns the filtered list of opportunities.

    [קלט]
    - profiles (dict): The complete dictionary of all available stocks 
    - prefs (dict): The dictionary containing the user's specific choices 

    [פלט]
    - dict: A filtered dictionary containing only the stocks that successfully passed the rules 
    """
    print("\n--- Running Rule-Based Filtering Logic ---")
    
    selected_stocks = {}
    
    # שליפת העדפות המשתמש
    target_domain = prefs.get("domain", "").strip().lower()
    target_risk = prefs.get("risk_level", "").strip().lower()
    
    # מעבר על כל מניה ובדיקת התאמה לחוקים
    for ticker, info in profiles.items():
        stock_domain = info.get("domain", "").strip().lower()
        stock_risk = info.get("risk_level", "").strip().lower()
        
        # חוק הסינון: התאמה מדויקת של תחום ורמת סיכון
        if stock_domain == target_domain and stock_risk == target_risk:
            selected_stocks[ticker] = info
            
    print(f"[V] Filtering complete. Found {len(selected_stocks)} matching stocks.")
    return selected_stocks

def generate_explanations_with_llm(preferences: dict, selected_stocks: dict) -> dict:
    """
    [הסבר על הפונקציה]
    Uses the Gemini API to generate personalized, educational rationales for each chosen stock .

    [הסבר על מהלך הפונקציה - איך הוא פועלת]
    1. Initializes the official Google GenAI client framework.
    2. Iterates over the selected stocks one by one 
    3. Constructs a detailed, dynamic text prompt combining the specific stock's data and user preferences 
    4. Sends the prompt over the network to the 'gemini-2.5-flash' model 
    5. Extracts the raw response text and saves it mapped to the stock's ticker 

    [קלט]
    - preferences (dict): User choices used to contextualize the explanation .
    - selected_stocks (dict): The filtered subset of stocks requiring explanations 

    [פלט]
    - dict: A mapping dictionary where keys are Tickers and values are the generated AI explanations 
    """
    print("\n--- Generating Educational Explanations with Gemini API ---")
    
    client = genai.Client()
    explanations = {}
    
    # שליפת העדפות המשתמש לצורך שילוב בפרומפט
    user_risk = preferences.get("risk_level")
    user_horizon = preferences.get("horizon")
    user_domain = preferences.get("domain")

    # מעבר על כל מניה שנבחרה ויצירת הסבר מותאם אישית
    for ticker, info in selected_stocks.items():
        print(f"Generating explanation for {ticker}...")
        
        # בניית פרומפט מובנה ומבוקר (Controlled Generation)
        prompt = f"""
        You are a helpful investment assistant. 
        Provide a short, concise, and educational explanation (max 3 sentences) explaining why the stock '{ticker}' matches the user's preferences.
        
        User Preferences:
        - Preferred Domain: {user_domain}
        - Desired Risk Level: {user_risk}
        - Investment Horizon: {user_horizon}
        
        Stock Profile:
        - Domain: {info.get('domain')}
        - Risk Level: {info.get('risk_level')}
        - Market Outlook: {info.get('outlook')}
        - Key Risks: {', '.join(info.get('key_risks', []))}
        
        Requirements:
        1. Explain how the stock's outlook aligns with the user's investment horizon.
        2. Mention at least one key risk the user should keep in mind, matching their chosen risk profile.
        3. Keep the tone professional, objective, and educational. Do not give financial advice to buy or sell.
        4. Answer in clear, plain English.
        """
        
        try:
            # פנייה ל ג'מיני
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            # שמירת התשובה במילון תחת סימול המניה
            explanations[ticker] = response.text.strip()
            
        except Exception as e:
            print(f"[X] Error generating explanation for {ticker}: {e}")
            explanations[ticker] = "Error: Could not generate explanation via AI API."
            
    return explanations

def main():
    """
    [הסבר על הפוקנציה]
    The central coordinator and entry point that runs the investment assistant pipeline end-to-end 

    [פלואו של הפונקציה הראשית]
    1. Triggers data loading to get stock statistics 
    2. Triggers the interactive console questions to get user needs 
    3. Runs the deterministic filtering algorithm on the data based on those needs 
    4. Passes the winners to the LLM component to generate tailored reports .
    5. Iterates through the results and cleanly prints the final dynamic dashboard 

    [קלט]
    - None.

    [פלט של הפעולה]
    - None (Executes workflow and outputs text to terminal) 
    """
    profiles = load_stock_profiles()
    prefs = ask_user_preferences()
    selected = filter_stocks(profiles, prefs)
    explanations = generate_explanations_with_llm(prefs, selected)
    
    print("\n=== Suggested Stocks ===\n")
    for ticker, explanation in explanations.items():
        print(f"- {ticker}:")
        print(f"  {explanation}\n")

if __name__ == "__main__":
    main()