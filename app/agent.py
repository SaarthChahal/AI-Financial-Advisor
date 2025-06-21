# 1. Basic Agent
from google.adk.agents import Agent

basic_agent = Agent(
    name="basic_agent",
    model="gemini-2.0-flash",
    description="A simple agent that answers questions",
    instruction="""
    You are a helpful stock market assistant. Be concise.
    If you don't know something, just say so.
    """,
)


# 2. Basic Agent with Tool
from google.adk.agents import Agent
import yfinance as yf

def get_stock_price(ticker: str):
    stock = yf.Ticker(ticker)
    price = stock.info.get("currentPrice", "Price not available")
    return {"price": price, "ticker": ticker}

tool_agent = Agent(
    name="tool_agent",
    model="gemini-2.0-flash",
    description="A simple agent that gets stock prices",
    instruction="""
    You are a stock price assistant. Always use the get_stock_price tool.
    Include the ticker symbol in your response.
    """,
    tools=[get_stock_price],
)


# 3. Agent with State
from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
import yfinance as yf

def get_stock_price(ticker: str, tool_context: ToolContext):
    stock = yf.Ticker(ticker)
    price = stock.info.get("currentPrice", "Price not available")
    
    # Initialize recent_searches if it doesn't exist
    if "recent_searches" not in tool_context.state:
        tool_context.state["recent_searches"] = []
        
    recent_searches = tool_context.state["recent_searches"]
    if ticker not in recent_searches:
        recent_searches.append(ticker)
        tool_context.state["recent_searches"] = recent_searches
    
    return {"price": price, "ticker": ticker}

stateful_agent = Agent(
    name="stateful_agent",
    model="gemini-2.0-flash",
    description="An agent that remembers recent searches",
    instruction="""
    You are a stock price assistant. Use the get_stock_price tool.
    I'll remember your previous searches and can tell you about them if you ask.
    """,
    tools=[get_stock_price],
)


# 4. Multi-Tool Agent
from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
import yfinance as yf

def get_stock_price(ticker: str, tool_context: ToolContext):
    stock = yf.Ticker(ticker)
    price = stock.info.get("currentPrice", "Price not available")
    
    # Initialize recent_searches if it doesn't exist
    if "recent_searches" not in tool_context.state:
        tool_context.state["recent_searches"] = []
        
    recent_searches = tool_context.state["recent_searches"]
    if ticker not in recent_searches:
        recent_searches.append(ticker)
        tool_context.state["recent_searches"] = recent_searches
    
    return {"price": price, "ticker": ticker}

def get_stock_info(ticker: str):
    stock = yf.Ticker(ticker)
    company_name = stock.info.get("shortName", "Name not available")
    sector = stock.info.get("sector", "Sector not available")
    return {
        "ticker": ticker,
        "company_name": company_name,
        "sector": sector
    }

multi_tool_agent = Agent(
    name="multi_tool_agent",
    model="gemini-2.0-flash",
    description="An agent with multiple stock information tools",
    instruction="""
    You are a stock information assistant. You have two tools:
    - get_stock_price: For prices
    - get_stock_info: For company name and sector
    """,
    tools=[get_stock_price, get_stock_info],
)


# 5. Structured Output Agent
from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field
import yfinance as yf

class StockAnalysis(BaseModel):
    ticker: str = Field(description="Stock symbol")
    recommendation: str = Field(description="Buy or Sell recommendation")

# Define a function to get stock data for our prompt
def get_stock_data_for_prompt(ticker):
    stock = yf.Ticker(ticker)
    price = stock.info.get("currentPrice", 0)
    target_price = stock.info.get("targetMeanPrice", 0)
    return price, target_price

structured_agent = LlmAgent(
    name="structured_agent",
    model="gemini-2.0-flash",
    description="An agent with structured output",
    instruction="""
    You are a stock advisor. Analyze the stock ticker provided by the user.
    Return Buy or Sell recommendation in JSON format.
    
    For each ticker, look at the price and target price to make a decision.
    If target price > current price: recommend Buy
    Otherwise: recommend Sell
    """,
    output_schema=StockAnalysis,
    output_key="stock_analysis"
)


# 6. Callback Agent
from google.adk.agents import Agent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool
import yfinance as yf
from typing import Dict, Any, Optional

def get_stock_data(ticker: str, tool_context: ToolContext):
    stock = yf.Ticker(ticker)
    price = stock.info.get("currentPrice", 0)
    
    # Initialize tool_usage in state if it doesn't exist
    if "tool_usage" not in tool_context.state:
        tool_context.state["tool_usage"] = {}
    
    return {
        "ticker": ticker,
        "price": price
    }

def before_tool_callback(tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext) -> Optional[Dict]:
    # Initialize tool_usage if it doesn't exist
    if "tool_usage" not in tool_context.state:
        tool_context.state["tool_usage"] = {}
        
    # Track tool usage count
    tool_usage = tool_context.state["tool_usage"]
    tool_name = tool.name
    tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
    tool_context.state["tool_usage"] = tool_usage
    
    print(f"[LOG] Running tool: {tool_name}")
    return None

def after_tool_callback(tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Dict) -> Optional[Dict]:
    print(f"[LOG] Tool {tool.name} completed")
    return None

# Initialize state before creating the agent
initial_state = {"tool_usage": {}}

callback_agent = Agent(
    name="callback_agent",
    model="gemini-2.0-flash",
    description="An agent with callbacks",
    instruction="""
    You are a stock assistant. Use get_stock_data tool to check stock prices.
    This agent keeps track of how many times tools have been used.
    """,
    tools=[get_stock_data],
    before_tool_callback=before_tool_callback,
    after_tool_callback=after_tool_callback,
)

# --- Tool-driven functions ---
from google.adk.agents import BaseAgent, Agent  
from google.genai import types 
from google.adk.events import Event 

def get_top_stocks_by_sector(sector: str = "Technology"):
    tickers_map = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMD", "AVGO"],
        "Banking": ["JPM", "BAC", "WFC", "GS", "C", "MS"],
    }
    stocks = tickers_map.get(sector, [])
    returns = []
    for t in stocks:
        hist = yf.Ticker(t).history(period="5y")
        if not hist.empty:
            ret = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
            returns.append({"ticker": t, "5y_return": round(ret,2)})
    top = sorted(returns, key=lambda x: x["5y_return"], reverse=True)[:6]
    tickers = [s["ticker"] for s in top]
    return {"sector": sector, "top_tickers": tickers, "5y_returns": {s["ticker"]: s["5y_return"] for s in top}}

def get_market_data(top_tickers: list):
    data = []
    for t in top_tickers:
        stk = yf.Ticker(t)
        info = stk.info
        hist = stk.history(period="1y")
        start = hist["Close"].iloc[0] if not hist.empty else None
        end = hist["Close"].iloc[-1] if not hist.empty else None
        one_year_return = round((end / start - 1)*100,2) if start and end else None
        data.append({
            "ticker": t,
            "price": info.get("currentPrice"),
            "beta": info.get("beta", 1.0),
            "1y_return": one_year_return
        })
    return {"market_data": data}

def evaluate_risk(market_data: list):
    risks = []
    for s in market_data:
        b = s["beta"]
        score = "Low" if b < 1 else "Medium" if b < 1.5 else "High"
        risks.append({"ticker": s["ticker"], "beta": b, "risk": score})
    return {"risk_scores": risks}

def build_portfolio(risk_scores: list, market_data: list, total_amount: float = 10000):
    weight = {"Low": 0.5, "Medium": 0.3, "High": 0.2}
    buckets = {"Low": [], "Medium": [], "High": []}
    md = {s["ticker"]: s for s in market_data}
    for r in risk_scores:
        buckets[r["risk"]].append(r["ticker"])

    alloc, exp_ret, stop = {}, 0, {}
    for risk_level, tickers in buckets.items():
        if not tickers: continue
        alloc_amt = weight[risk_level] * total_amount / len(tickers)
        for t in tickers:
            alloc[t] = round(alloc_amt,2)
            yr = md[t].get("1y_return") or 0
            exp_ret += alloc_amt * yr / 100
            stop[t] = round(md[t]["price"] * 0.85,2)

    return {"allocation": alloc, "expected_1y_return": round(exp_ret,2), "stop_losses": stop}

# --- Wrapping each step in Tool-Like Custom Agents ---

class TopPickerAgent(BaseAgent):
    async def _run_async_impl(self, ctx):
        sector = ctx.session.state.get("user_sector", "Technology")
        out = get_top_stocks_by_sector(sector)
        ctx.session.state["top_stocks"] = out  # Save to context
        
        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=str(out))])
        )


class MarketDataAgent(BaseAgent):
    async def _run_async_impl(self, ctx):
        top = ctx.session.state["top_stocks"]
        tickers = top["top_tickers"]
        out = get_market_data(tickers) 
        ctx.session.state["market_data"] = out["market_data"]  # Save to context 
        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=str(out))])
        )


class RiskAgent(BaseAgent):
    async def _run_async_impl(self, ctx):
        md = ctx.session.state["market_data"]
        out = evaluate_risk(md) 
        ctx.session.state["risk_scores"] = out  # Save to context 
        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=str(out))])
        )

class PortfolioAgent(BaseAgent):
    async def _run_async_impl(self, ctx):
        rs_wrapped = ctx.session.state["risk_scores"]
        rs = rs_wrapped["risk_scores"] 

        md = ctx.session.state["market_data"]
        out = build_portfolio(rs, md, ctx.session.state.get("investment",10000))
        ctx.session.state["portfolio"] = out  # Save to context
        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=str(out))])
        )
from google.adk.agents import SequentialAgent 

investment_advisor_agent = SequentialAgent( 
    name="investment_advisor",
    sub_agents=[
        TopPickerAgent(name="picker"),
        MarketDataAgent(name="market"),
        RiskAgent(name="risk"),
        PortfolioAgent(name="portfolio")
    ]
)



# # --- Determine Which Agent to Use ---

from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext, FunctionTool
from google.adk.runners import Runner
from google.genai import types
 

import re

# --- Helper: Extract investment amount ---
def extract_investment_amount(prompt: str):
    match = re.search(r"\$?\s?(\d{3,})", prompt)
    return float(match.group(1)) if match else None

def extract_ticker_or_company(prompt: str) -> str:
    """
    Attempts to extract a known stock ticker from user input.
    Falls back to uppercased string if not recognized.
    """
    known_mappings = {
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "apple": "AAPL",
        "microsoft": "MSFT",
        "meta": "META",
        "facebook": "META",
        "amazon": "AMZN",
        "nvidia": "NVDA",
        "tesla": "TSLA",
        "netflix": "NFLX",
        "broadcom": "AVGO",
        "amd": "AMD"
    }

    prompt_lower = prompt.lower()
    for name, ticker in known_mappings.items():
        if name in prompt_lower:
            return ticker

    # Fallback: assume the prompt is already a ticker or shorthand
    return prompt.strip().upper()



# # --- Tool for routing logic ---
# def dynamic_router(prompt: str, tool_context: ToolContext):
#     """Routes based on presence of investment amount."""
#     amount = extract_investment_amount(prompt)
#     if amount:
#         tool_context.state["investment"] = amount
#         tool_context.actions.transfer_to_agent = "structured_agent"
#         return f"Transferring to investment planner (amount: {amount})"
#     else:
#         tool_context.state["ticker"] = prompt.strip().upper()
#         tool_context.actions.transfer_to_agent = "multi_tool_agent"
#         return f"Looking up stock info for '{prompt.strip().upper()}'..."
    


def dynamic_router(prompt: str, tool_context: ToolContext) -> str:
    """
    Routes user query dynamically:
    - If it includes an investment amount: transfer to SequentialAgent.
    - If it includes a ticker + advisory intent: transfer to structured_agent.
    - If it includes only a ticker/company: transfer to multi_tool_agent.
    """
    prompt_lower = prompt.lower()
    amount = extract_investment_amount(prompt)

    # Case 1: Investment amount detected → SequentialAgent
    if any(term in prompt_lower for term in ["should i", "buy", "sell", "recommendation", "invest", "advice"]) and amount:
        tool_context.state["investment"] = amount
        tool_context.actions.transfer_to_agent = "investment_advisor"
        return f"Routing to full investment planning with ${amount}."

    # Case 2: Advice-related terms + ticker/company name → structured_agent
    elif any(term in prompt_lower for term in ["should i", "buy", "sell", "recommendation", "invest", "advice"]):
       # tool_context.state["ticker"] = prompt.strip().upper()
        tool_context.state["ticker"] = extract_ticker_or_company(prompt)
        tool_context.actions.transfer_to_agent = "structured_agent"
        #return f"Analyzing '{prompt.strip().upper()}' for investment advice."
        return f"Analyzing '{tool_context.state['ticker']}' for investment advice." 

    # Case 3: Only a stock/company name → multi_tool_agent
    else:
        tool_context.state["ticker"] = prompt.strip().upper()
        tool_context.actions.transfer_to_agent = "multi_tool_agent"
        return f"Fetching stock info for '{prompt.strip().upper()}'..."




# --- Router Agent (root agent) ---
router_tool = FunctionTool(func=dynamic_router)

root_agent = Agent(
    model="gemini-2.0-flash",
    name="root_router",
    instruction="""You are a router agent. When a user gives a prompt, use the dynamic_router tool to decide which agent should handle it next.""",
    tools=[router_tool]
)


# --- Attach sub-agents to root agent ---
root_agent.sub_agents = [structured_agent, multi_tool_agent,investment_advisor_agent]

# --- ADK needs this for auto-import ---
# DO NOT REMOVE THIS EXPORT
__all__ = ["root_agent"]
