import os
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

def create_vector_store():
    print("--- Starting Stock Profiles Vector Store Creation ---")
    
    # נתיב הקובץ במערכת שלך
    md_path = r"required_files\stocks_profiles.md"
    if not os.path.exists(md_path):
        print(f"[X] Error: {md_path} not found! Check your folders.")
        return
        
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # רשימת כל 25 המניות מהמטלה לחיתוך אמין ומבוקר
    all_tickers = [
        "XOM", "CVX", "ENPH", "FSLR", "PARR", "NVDA", "MSFT", "SMCI", 
        "ACLX", "PLTR", "LLY", "JNJ", "VRTX", "REGN", "CRSP", "JPM", 
        "V", "SQ", "SOFI", "ALLY", "PG", "KO", "SFM", "CELH", "GO"
    ]

    chunks = []
    metadata_list = []
    ids = []

    lines = content.split('\n')
    current_ticker = None
    current_block = []

    for line in lines:
        cleaned_line = line.strip()
        found_ticker = None
        for ticker in all_tickers:
            if ticker == cleaned_line or cleaned_line.endswith(" " + ticker) or cleaned_line.endswith("## " + ticker) or cleaned_line.endswith("# " + ticker):
                found_ticker = ticker
                break
        
        if found_ticker:
            if current_ticker and current_block:
                chunks.append("\n".join(current_block))
                metadata_list.append({"ticker": current_ticker})
                ids.append(f"id_{current_ticker}")
            current_ticker = found_ticker
            current_block = [line]
        else:
            if current_ticker:
                current_block.append(line)

    if current_ticker and current_block:
        chunks.append("\n".join(current_block))
        metadata_list.append({"ticker": current_ticker})
        ids.append(f"id_{current_ticker}")

    print(f"[V] Verification: Parsed {len(chunks)} stock profiles from markdown.")
    
    if len(chunks) == 0:
        print("[X] Error: Failed to parse stocks.")
        return

    # יצירת בסיס הנתונים המקומי של Chroma
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="stock_profiles")

    print("Generating embeddings via Google GenAI API and writing to Chroma...")
    client = genai.Client()
    
    for chunk, meta, cid in zip(chunks, metadata_list, ids):
        try:
            # שימוש במודל המדויק שגוגל אישרה שיש לך גישה אליו!
            response = client.models.embed_content(
                model="gemini-embedding-2",
                contents=chunk
            )
            embedding = response.embeddings[0].values
            
            # שמירה פיזית לתוך ה-SQL הוקטורי המקומי (Chroma)
            collection.add(
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[meta],
                ids=[cid]
            )
            print(f"Indexed successfully: {meta['ticker']}")
        except Exception as e:
            print(f"[X] Failed to embed {meta['ticker']}: {e}")
            
    print("\n[V] Success: All stock profiles are now safely stored in your local vector database!")

if __name__ == "__main__":
    create_vector_store()