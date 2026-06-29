from google import genai
from dotenv import load_dotenv

load_dotenv()

###
# בדקנו אילו מודלי של גוגלים זמינים שנוכל לפנות אליהם... בקוד מבוקר לבדיקה בלבד
###

print("--- Fetching All Available Models (Strict Mode) ---")
client = genai.Client()

try:
    # שליפת רשימת המודלים הכללית
    response = client.models.list()
    
    print("\n[V] Connected to Google successfully! Searching for embedding models...")
    found_any = False
    
    for model in response:
        # מדפיסים כל מודל שמכיל את המילה 'embed' בשם שלו
        if "embed" in model.name.lower():
            print(f" - Model Name: {model.name}")
            # נדפיס את השיטות הנתמכות בפורמט החדש של ה-SDK
            if hasattr(model, 'supported_generation_methods'):
                print(f"   Supported methods: {model.supported_generation_methods}")
            found_any = True
            
    if not found_any:
        print("[!] No explicit embedding models found in the quick scan.")
        print("Printing top 5 available models just for verification:")
        for i, model in enumerate(response):
            if i < 5:
                print(f" - {model.name}")

except Exception as e:
    print(f"[X] Critical Error during interrogation: {e}")