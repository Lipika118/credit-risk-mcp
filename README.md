# Credit Risk Analytics MCP Server

A Python MCP (Model Context Protocol) server that exposes credit-risk analytics
as callable tools for Claude Desktop — turning natural-language questions into
real financial risk calculations.

## What it does

This server gives Claude three tools:

| Tool | What it does |
|---|---|
| `get_company_financials` | Pulls live balance sheet & income statement data for any stock ticker (via `yfinance`) |
| `calculate_altman_zscore` | Computes the Altman Z-Score — a classic bankruptcy-risk formula combining 5 financial ratios — for a public company |
| `predict_loan_default_risk` | Predicts an individual loan applicant's default probability using a logistic regression model |

Ask Claude Desktop something like *"What's the Altman Z-Score for TCS.NS?"*
or give it a loan applicant's income, debt ratio, and credit history, and it
calls the right tool, runs the real calculation, and explains the result.

## Why MCP

Without MCP, these would just be Python functions you'd have to run yourself.
MCP turns them into tools an AI client can call directly: Claude Desktop sends
a structured JSON-RPC request to this server, the server runs the actual
calculation, and sends the result back — so you get a live, verifiable answer
instead of a guess from the model's training data.

## Project structure

```
credit-risk-mcp/
├── server.py          # The MCP server — defines all 3 tools
├── train_model.py      # Generates synthetic credit data + trains the logistic regression model
├── requirements.txt    # Python dependencies
├── model.pkl            # Trained logistic regression model
├── scaler.pkl            # StandardScaler used to preprocess model inputs
└── .gitignore
```

## Setup

```bash
git clone https://github.com/Lipika118/credit-risk-mcp.git
cd credit-risk-mcp
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The trained model (`model.pkl`, `scaler.pkl`) is already included, so you can
skip straight to running the server. If you want to retrain it yourself:

```bash
python3 train_model.py
```

This generates a synthetic-but-realistic applicant dataset (income, debt
ratio, credit history, late payments, loan amount, age), trains a logistic
regression model, and prints the test AUC.

## Testing standalone

Before connecting to Claude Desktop, test the tools directly with the MCP
Inspector:

```bash
pip install "mcp[cli]"
mcp dev server.py
```

This opens a browser UI where you can call each tool manually and see the
JSON-RPC request/response for each one.

## Connecting to Claude Desktop

Add this to your `claude_desktop_config.json`
(`%APPDATA%\Claude\claude_desktop_config.json` on Windows,
`~/Library/Application Support/Claude/claude_desktop_config.json` on Mac):

```json
{
  "mcpServers": {
    "credit-risk": {
      "command": "/full/path/to/venv/Scripts/python.exe",
      "args": ["/full/path/to/credit-risk-mcp/server.py"]
    }
  }
}
```

Fully quit and reopen Claude Desktop, then check **Connectors** in the chat
input menu — `credit-risk` should be listed and toggled on.

## Example usage

**Company risk:**
> "What's the Altman Z-Score for TCS.NS?"

```
Ticker: TCS.NS
Altman Z-Score: 10.69
Risk Zone: Safe zone (low bankruptcy risk)

Component ratios:
  Working Capital / Total Assets: 0.410
  Retained Earnings / Total Assets: 0.548
  EBIT / Total Assets: 0.366
  Market Cap / Total Liabilities: 11.271
  Revenue / Total Assets: ...
```

**Individual risk:**
> "A loan applicant has monthly income 40000, debt-to-income ratio 0.5,
> 3 years credit history, 2 late payments last year, wants a loan of
> 250000, and is 27 — what's their default risk?"

```
Default probability: 78.0%
Risk band: High risk
```

## Notes on the model

The loan-default model is trained on **synthetic data**, not real applicant
records — this avoids privacy/licensing issues while still learning
genuine, explainable relationships (higher debt-to-income ratio and more
late payments both increase predicted default risk). It's meant to
demonstrate the MCP integration pattern, not to be used for real lending
decisions.

## Safety

`predict_loan_default_risk` and `calculate_altman_zscore` are both
read-only — they don't modify any data or make external calls beyond
fetching public market data.

## License

MIT
