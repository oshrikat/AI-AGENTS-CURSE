# מטלה — שלב 3: גרסת RAG (retrieval-augmented generation) של הסוכן משלב 2 (agent_s3.py)

## סקירה קומפקטית

גרסה זו של הסוכן (ראו `agents_plots/agent_s3.png`) אמורה להיות גרסת ה־RAG (retrieval-augmented generation) של הסוכן משלב 2.
במקום מנגנון בחירת מניות המבוסס על regex, בחירת המניות מתבצעת באמצעות חיפוש דמיון (similarity search) ב־vector store שמכיל את פרופילי המניות.

## הכנה

- כדי לבצע RAG באמצעות vector store, קודם צריך לכתוב את נתוני המניות לתוך vector store. סוג אפשרי של vector store יכול להיות Chroma, אם כי אפשר להשתמש גם באפשרויות אחרות.

- כתבו תוכנית בשם **profiles_to_vecstore.py** שמקבלת את **stock_profiles.md** וכותבת אותם לתוך vector store. ודאו שכל chunk ב־vector store יתאים למניה אחת (ticker). אם לכל chunk ב־vector store יש metadata, ה־metadata צריך להיות ה־ticker, למשל `"MSFT"`.

- הריצו את התוכנית הזו.

## המשימה

שנו את `stock_selector_node` (מהסוכן של שלב 2). במקום טכניקת בחירת מניות המבוססת על regex, השתמשו בחיפוש ב־vector store.

ראשית, צרו query שכולל את העדפות המשתמש. לאחר מכן העבירו את ה־query לפונקציה שמבצעת similarity search ב־vector store שמכיל את פרופילי המניות.

בדרך כלל, ה־vector store מכיל מתודה כלשהי שמקבלת query כקלט ומבצעת similarity search.
