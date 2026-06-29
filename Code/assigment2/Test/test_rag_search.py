# קוד בדיקה שהקוד חלק א' עובד ותקין שומר ווקטורים והכל תקין  #

import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

print("--- Testing Semantic Similarity Search in Chroma ---")

# 1. התחברות למסד הנתונים הוקטורי הקיים על הדיסק
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="stock_profiles")

# 2. הגדרת שאילתת חיפוש חופשית (כמו שהמשתמש יבקש)
# נחפש משהו שקשור לאנרגיה וסיכון בינוני
query = "I want a medium risk company in the Energy sector"
print(f"Searching for: '{query}'")

try:
    # 3. המרת השאילתה של המשתמש לווקטור באותו מודל בדיוק!
    client = genai.Client()
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=query
    )
    query_embedding = response.embeddings[0].values

    # 4. ביצוע החיפוש הסמנטי בתוך Chroma (מבקשים את 2 התוצאות הכי קרובות)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    # 5. הדפסת המדדים והתוצאות
    print("\n=== SEARCH RESULTS ===")
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print(f"\n[Matched Stock Ticker]: {meta['ticker']}")
        print(f"Content found:\n{doc}")
        print("-" * 40)

except Exception as e:
    print(f"[X] Search failed: {e}")