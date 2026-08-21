"""
Credit Risk Analytics MCP Server
---------------------------------
Exposes company Altman Z-Score and individual loan default risk as
tools Claude Desktop can call.
"""

import warnings
warnings.filterwarnings("ignore")
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("credit-risk")

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    LOAN_MODEL = pickle.load(f)
with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    SCALER = pickle.load(f)

FEATURE_ORDER = [
    "annual_income", "debt_to_income_ratio", "credit_history_years",
    "late_payments_2yr", "loan_amount", "age",
]


@mcp.tool()
def get_company_financials(ticker: str) -> str:
    """Fetch key balance sheet and income statement figures for a stock ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        bs = stock.balance_sheet
        fin = stock.financials
        if bs.empty or fin.empty:
            return f"Could not retrieve financial statements for '{ticker}'."
        latest_bs = bs.iloc[:, 0]
        latest_fin = fin.iloc[:, 0]
        summary = {
            "Company": info.get("longName", ticker),
            "Market Cap": info.get("marketCap"),
            "Total Assets": latest_bs.get("Total Assets"),
            "Total Liabilities": latest_bs.get("Total Liabilities Net Minority Interest"),
            "Current Assets": latest_bs.get("Current Assets"),
            "Current Liabilities": latest_bs.get("Current Liabilities"),
            "Retained Earnings": latest_bs.get("Retained Earnings"),
            "EBIT": latest_fin.get("EBIT"),
            "Total Revenue": latest_fin.get("Total Revenue"),
        }
        return "\n".join(f"{k}: {v}" for k, v in summary.items())
    except Exception as e:
        return f"Error fetching data for '{ticker}': {e}"


@mcp.tool()
def calculate_altman_zscore(ticker: str) -> str:
    """
    Calculate the Altman Z-Score for a public company.
    Z > 2.99: Safe | 1.81-2.99: Grey | Z < 1.81: Distress
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        bs = stock.balance_sheet
        fin = stock.financials
        if bs.empty or fin.empty:
            return f"Could not retrieve financial statements for '{ticker}'."

        latest_bs = bs.iloc[:, 0]
        latest_fin = fin.iloc[:, 0]

        total_assets = latest_bs.get("Total Assets")
        total_liabilities = latest_bs.get("Total Liabilities Net Minority Interest")
        current_assets = latest_bs.get("Current Assets")
        current_liabilities = latest_bs.get("Current Liabilities")
        retained_earnings = latest_bs.get("Retained Earnings")
        ebit = latest_fin.get("EBIT")
        revenue = latest_fin.get("Total Revenue")
        market_cap = info.get("marketCap")

        if not all([total_assets, current_assets, current_liabilities,
                    retained_earnings, ebit, revenue, market_cap, total_liabilities]):
            return f"Missing required financial fields for '{ticker}' to compute Z-Score."

        working_capital = current_assets - current_liabilities
        X1 = working_capital / total_assets
        X2 = retained_earnings / total_assets
        X3 = ebit / total_assets
        X4 = market_cap / total_liabilities
        X5 = revenue / total_assets

        z_score = 1.2 * X1 + 1.4 * X2 + 3.3 * X3 + 0.6 * X4 + 1.0 * X5

        if z_score > 2.99:
            zone = "Safe zone (low bankruptcy risk)"
        elif z_score > 1.81:
            zone = "Grey zone (moderate risk, monitor closely)"
        else:
            zone = "Distress zone (high bankruptcy risk)"

        return (
            f"Ticker: {ticker}\nAltman Z-Score: {z_score:.2f}\nRisk Zone: {zone}\n\n"
            f"Component ratios:\n"
            f"  Working Capital / Total Assets: {X1:.3f}\n"
            f"  Retained Earnings / Total Assets: {X2:.3f}\n"
            f"  EBIT / Total Assets: {X3:.3f}\n"
            f"  Market Cap / Total Liabilities: {X4:.3f}\n"
            f"  Revenue / Total Assets: {X5:.3f}"
        )
    except Exception as e:
        return f"Error calculating Z-Score for '{ticker}': {e}"


@mcp.tool()
def predict_loan_default_risk(
    annual_income: float,
    debt_to_income_ratio: float,
    credit_history_years: float,
    late_payments_2yr: int,
    loan_amount: float,
    age: int,
) -> str:
    """Predict probability that a loan applicant will default."""
    features = pd.DataFrame([[
        annual_income, debt_to_income_ratio, credit_history_years,
        late_payments_2yr, loan_amount, age,
    ]], columns=FEATURE_ORDER)

    scaled = SCALER.transform(features)
    default_prob = LOAN_MODEL.predict_proba(scaled)[0][1]

    if default_prob < 0.10:
        band = "Low risk"
    elif default_prob < 0.30:
        band = "Moderate risk"
    else:
        band = "High risk"

    return (
        f"Default probability: {default_prob:.1%}\nRisk band: {band}\n\n"
        f"Applicant profile:\n"
        f"  Annual income: {annual_income}\n"
        f"  Debt-to-income ratio: {debt_to_income_ratio}\n"
        f"  Credit history: {credit_history_years} years\n"
        f"  Late payments (last 2 years): {late_payments_2yr}\n"
        f"  Loan amount requested: {loan_amount}\n"
        f"  Age: {age}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")