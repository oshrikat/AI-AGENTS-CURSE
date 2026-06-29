# Assignment — (agent_s3.py): RAG (retrieval-augmented generation) version of stage 2 agent

## Compact Overview
This version of the agent (see agents_plots/agent_s3.png) is supposed to be the RAG (retrieval-augmented generation) version of  agent_s2.py.
Instead of regex-based stock selection mechanism, the stock selection is done via similarity search on the vector store (containing the stock profiles).

## Preparation
- In order to do RAG using a vector store we first need to write the stock data to a vector store. Possible type of vector store could be Chroma, although other variants are possible too.

- write a program **profiles_to_vecstore.py** that takes **stock_profiles.md** and writes them to a vector store. make sure that each chink in vector store will correspond to one stock (ticker). if each chunk in vector store has metadata, the metadata should be the ticker ("MSFT").

- run this program

## Task
Modify stock_selector_node (from [agent_s2]). Instead of regex-based technique of stock selection, use vector store search. First create query that includes the user's preferences. Then pass this query to some function that does similarity search on the vector store (that contains the stock profiles).
Usually the vector store contains some method that takes query as input and does similarity search.

## Test Scenarios
Perform the following test scenarious. Save all tests to a docx file (for each test, a record of user-agent interaction). Name of file: tests_[assignment_nr].docx
(e.g. tests_3.docx)

### Scenario 1 — Normal happy path (tech, medium risk, long horizon)

**User input** (interactive CLI):

Domain prompt → tech

Risk prompt → medium

Horizon prompt → long (5y+)

**Expected agent response** (abstract):

Agent prints confirmation that it collected preferences.

Agent prints a DEBUG: Retrieved chunks section with up to 4 chunks that are semantically related to domain: tech, risk level: medium, horizon: long (5y+).

Final answer is a friendly paragraph + bullet points that:

Names up to 3 concrete stocks that appear in the retrieved chunks.

Gives 2–3 sentences per stock explaining why each fits tech + medium risk + long horizon.

Does not recommend stocks that are not mentioned in the retrieved chunks.

### Scenario 2 — “No good match” / cannot identify suitable stocks

**User input** (interactive CLI):

Domain prompt → energy

Risk prompt → low

Horizon prompt → short 0–2y

**Expected agent response** (abstract):

Agent retrieves chunks (still prints the debug preview), but:

Either the chunks are weakly related / off-domain, or

They don’t contain identifiable stock names relevant to the criteria.

Final answer should follow the prompt instruction and explicitly say something equivalent to:
“Sorry, none of the retrieved information matches your criteria well enough.”

No fabricated stock names; if it lists stocks, they must be from retrieved chunks.

### Scenario 3 — Input validation loops (wrong domain and risk first)

**User input** (interactive CLI):

Domain prompt → biotech (invalid)

Domain prompt again → health (valid)

Risk prompt → moderate (invalid)

Risk prompt again → high (valid)

Horizon prompt → medium 2–5y

**Expected agent response** (abstract):

After biotech, agent prints the guidance message:
“choose from tech, health, finance, energy, consumer_goods”

After moderate, agent prints:
“Please type 'low', 'medium', or 'high'.”

Once valid inputs are provided, agent proceeds normally:

Prints debug retrieved chunks (top 4).

Final answer: bullet-point shortlist of up to 3 stocks from retrieved chunks, each justified in 2–3 sentences, aligned to health + high risk + medium horizon.

