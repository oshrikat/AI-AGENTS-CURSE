# Assignment — Stage 1: Simple Compact non-agentic interaction:  Generate stock explanation using LLM (Large Language model) API

## Overview of the new technology: LLM APIs
In this stage you will integrate a **Large Language Model (LLM) API** into an otherwise deterministic program. An LLM API works like a remote function call: you send structured text input (a prompt or messages), and the service returns generated text. The key idea in this assignment is **controlled generation** — using prompt design to constrain the model so it explains results rather than making decisions.

## Goal
Create a program that selects a few stocks (tickers) based on user preferences and provides short explanation for each stock gnerated by LLM.

## Task

You are given the main() program whose.  
Your task is to write the agent so it works end-to-end as an **investment assistant**.

At a high level, the agent should:
1. Collect user investment preferences  
2. Select suitable stocks from predefined profiles  
3. Generate short, educational explanations for the selected stocks using an LLM  
4. Present the results to the user  

---

## Required Functions

Implement ** the following functions**, with the same names and roles as in the provided code.

---
## 1.load_stock_profiles()
input: json file
### Description
Tiny function that loads json file into dictionary


## 2. ask_user_preferences()

def ask_user_preferences() -> dict:

### Description

This function is responsible for collecting the user’s investment preferences.

### Responsibilities

Ask the user for:

preferred investment domain

risk tolerance

investment horizon

Return the preferences in a structured dictionary format that can be used by the rest of the agent.

## 3. filter_stocks_by_preferences()
def filter_stocks_by_preferences(
    profiles: dict,
    prefs: dict,
    min_count: int = 3,
    max_count: int = 5
) -> dict:
### Description

This function selects a small set of stocks that best match the user’s preferences.

### Responsibilities

Receive the full stock profile collection and the user preferences

Apply rule-based logic to choose an appropriate subset of stocks

Return a dictionary of selected stock profiles keyed by ticker symbol


## 4. generate_explanations_with_llm()
def generate_explanations_with_llm(
    preferences: dict,
    selected_stocks: dict
) -> dict:

### Description

This function uses a Large Language Model (LLM) to generate short explanations for each selected stock.

### Responsibilities

Build a prompt that includes user preferences and relevant information from each stock profile.

Call the LLM API to generate explanations

Return a dictionary mapping each stock ticker to its explanation text

## 5. main function (provided)
```
def main():
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
```






