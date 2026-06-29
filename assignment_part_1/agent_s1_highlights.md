# Code Highlights: LLM API Usage

Topic: Introduction to LLM APIs in Python 


---

## Highlight 1: API Authentication

**Concepts illustrated:**
- Environment variables for API key management
- Client instantiation

**Code snippet:**

```python
import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = openai_key
client = OpenAI()
```

---

## Highlight 2: Structured Prompt Construction

**Concepts illustrated:**
- prompt for LLM
    - prompt templating: prompt contains variables ( e.g. {ticker})
    - Encoding user preferences into prompts

**Code snippet:**

```python
prompt = f"""
You are an investment education assistant.

User preferences:
- Domain: {preferences['domain']}
- Risk tolerance: {preferences['risk_tolerance']}
- Investment horizon: {preferences['horizon']}

Stock:
- Ticker: {ticker}

Task:
Explain in 2–3 short sentences why this stock matches the user's preferences.
"""
```

---

## Highlight 3: Model Invocation

**Concepts illustrated:**
- obtaining response from LLM
    - prompt is input
    - Model selection as configurable parameter
    - Message list format


**Code snippet:**

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
```

---

## Highlight 4: Response Extraction

**Concepts illustrated:**
- Structured API response object
- Accessing generated text from response
- Post-processing extracted text

**Code snippet:**

```python
explanations[ticker] = response.choices[0].message.content.strip()
```

---


