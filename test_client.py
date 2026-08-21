import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run():
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"]
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # AAPL
            res_aapl = await session.call_tool("calculate_altman_zscore", arguments={"ticker": "AAPL"})
            print("=== AAPL RESULT ===")
            print(res_aapl.model_dump_json(indent=2))
            
            # Distressed company (AMC)
            res_nkla = await session.call_tool("calculate_altman_zscore", arguments={"ticker": "AMC"})
            print("=== DISTRESSED RESULT (AMC) ===")
            print(res_nkla.model_dump_json(indent=2))

            # Risky Profile (25000 income -> converted to 25k since the signature does not specify 25k versus 2500, but in train_model monthly income mean is 8000. So 25000 is high but it's what they asked)
            res_risky = await session.call_tool("predict_loan_default_risk", arguments={
                "annual_income": 25000,
                "debt_to_income_ratio": 0.9,
                "credit_history_years": 1,
                "late_payments_2yr": 6,
                "loan_amount": 30000,
                "age": 24
            })
            print("=== RISKY PROFILE ===")
            print(res_risky.model_dump_json(indent=2))

            # Safe Profile
            res_safe = await session.call_tool("predict_loan_default_risk", arguments={
                "annual_income": 120000,
                "debt_to_income_ratio": 0.15,
                "credit_history_years": 15,
                "late_payments_2yr": 0,
                "loan_amount": 10000,
                "age": 40
            })
            print("=== SAFE PROFILE ===")
            print(res_safe.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(run())
