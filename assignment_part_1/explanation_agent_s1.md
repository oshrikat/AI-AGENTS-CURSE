# Compact Overview (Stage 1: non-agentic LLM API )

This first version is a minimal investment consultant whose sole purpose is to demonstrate basic **LLM API** usage in a concrete application. It combines simple, deterministic rule-based stock filtering with an LLM that is used only to generate natural-language explanations, not to make decisions. The program is fully stateless and synchronous, with no agent loop, memory, or tool orchestration. **Prompting is explicit and constrained**, emphasizing an educational tone and avoiding financial advice. Overall, it serves as a clean baseline before introducing agentic concepts such as planning, memory, and tool use.

# Explanation of the Core Technology Large Language Model (LLM) APIs 
Large Language Model APIs provide programmatic access to advanced language systems such as ChatGPT, Claude, and Gemini, allowing software to generate, analyze, and transform text on demand. These APIs expose the models as remote services that accept structured inputs (prompts or message lists) and return textual outputs, making them easy to integrate into existing applications. From a developer’s perspective, an LLM API behaves like a stateless function call: each request is handled independently, with no built-in memory unless the application explicitly supplies prior context.
By using an API, developers are abstracted away from model hosting, scaling, and operational complexity, and instead work with high-level controls such as model choice, response length, and output style. This enables rapid experimentation and iteration, since changing system behavior often requires only modifying the prompt rather than rewriting logic. In practice, LLM APIs turn natural language into a flexible interface layer, where systems can delegate explanation, summarization, and reasoning-like tasks to models while retaining full control in conventional code.

How LLM API is used in this program
In this program, the LLM API is integrated in a **clear, linear, and inspectable manner**, making it suitable for demonstrating raw API usage. The interaction begins with configuring authentication by assigning the API key to an environment variable using `os.environ["OPENAI_API_KEY"] = openai_key`, followed by instantiating the client with `client = OpenAI()`. This setup phase shows how access to an external LLM service is established before any model calls are made.

For each stock selected by the rule-based logic, the program constructs a **structured natural-language prompt** using a formatted multiline string, assigned via `prompt = f""" ... """`. The prompt embeds user preferences, stock attributes, and explicit behavioral constraints, effectively turning natural language into a control mechanism for the model’s output. This highlights how prompt text itself becomes a programmable interface.

The actual model invocation occurs in the call to `client.chat.completions.create(...)`, where the model identifier (`"gpt-4o-mini"`) and the user message are supplied. The API returns a structured response object, from which the generated explanation is extracted using `response.choices[0].message.content`. The extracted text is then post-processed, stored, and printed, completing a full **request–response–integration cycle** in which the LLM acts as a controlled language generation component within a deterministic Python pipeline.




#  Usage of the Core Technology (LLM API) in this code

- `os.environ["OPENAI_API_KEY"] = openai_key`  
  Injects credentials via environment variables, following standard API authentication practice.

- `client = OpenAI()`  
  Creates a client object that encapsulates all interaction with the LLM service.

- `prompt = f""" ... """`  
  Constructs a structured natural-language prompt that encodes context, data, and behavioral constraints.

- `client.chat.completions.create(...)`  
  Performs the core API call: sends the prompt to the LLM and requests a generated response.

- `"model": "gpt-4o-mini"`  
  Explicitly selects the model, making model choice a configurable part of the application.

- `response.choices[0].message.content`  
  Extracts the generated text from the structured API response for downstream use.
  
# Test Scenarios

## Test Scenario 1 — “Strict matches exist” (domain + risk satisfied)

**User input** (3 prompts):

Preferred domain: Technology

Risk tolerance: medium

Investment horizon: long term

**Expected agent response** (structure, not exact tickers/text):

Prints === Suggested Stocks ===

Outputs 3–5 tickers whose profiles match:

domain contains “Technology” (case-insensitive substring match)

risk_level is exactly medium

For each ticker, prints 2–3 short educational sentences explaining the match (domain + medium risk + long-term framing), and explicitly avoids financial advice.

Ends with the built-in disclaimer paragraph.



## Test Scenario 2 — “Too few strict matches → risk constraint relaxed” (domain-only fallback)

**User input** (3 prompts):

Preferred domain: Healthcare

Risk tolerance: low

Investment horizon: short term

**Expected agent response** (structure, not exact tickers/text):

Prints === Suggested Stocks ===

Because fewer than 3 strict matches are found (domain + low risk), the agent relaxes risk filtering and selects 3–5 Healthcare tickers of mixed risk levels (still domain-matched).

Each ticker gets a 2–3 sentence explanation that:

Still references the user’s Healthcare preference

Acknowledges risk tolerance context in an educational way (e.g., “this may be higher risk than requested” / “risk differs”)

Keeps “educational, not advisory” tone.

Ends with the built-in disclaimer paragraph.  

