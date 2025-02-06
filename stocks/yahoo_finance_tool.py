from typing import Type, Union
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import yfinance as yf

class MyToolInput(BaseModel):
    ticket: str

class YahooFinanceTool(BaseTool):
    name: str = "Yahoo Finance Tool"
    description: str = "Fetches stock prices for {ticket} from the last year about a specific company from the Yahoo Finance API"
    args_schema: Type[BaseModel] = MyToolInput
    
    def fetch_stock_price(self, ticket: str):
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365)
        stock = yf.download(ticket, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        return stock

    def _run(self, ticket: Union[str, dict]) -> str:
        if isinstance(ticket, dict) and "description" in ticket:
            ticket = ticket["description"]
        
        # Ensure ticket is now a string before validation
        if not isinstance(ticket, str):
            raise ValueError("Invalid ticket format. Expected a string.")

        return self.fetch_stock_price(ticket)
