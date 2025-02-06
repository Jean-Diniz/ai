import os
from datetime import datetime
from crewai import Agent, Task, Crew, Process, LLM
from langchain_community.tools import DuckDuckGoSearchResults
from crewai_tools import CSVSearchTool
from dotenv import load_dotenv
from yahoo_finance_tool import YahooFinanceTool


load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')

llmLocal = LLM(
    model="ollama/deepseek-r1:1.5b",
    temperature=0.1,
    base_url="http://localhost:11434"
)
llm = LLM(
    model="groq/gemma2-9b-it",
    temperature=0.1,
    max_retries=2,
)
csvWalletTool = CSVSearchTool(
    csv="Wallet.csv",
    config=dict(
        llm=dict(
            provider="groq", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama3-8b-8192",
                temperature=0.7,
            ),
        ),
        embedder=dict(
            provider="ollama",
            config=dict(
                model="nomic-embed-text"
            ),
        ),
    )
)

customerManager = Agent(
    role="Customer Stocks Manager",
    goal= "Get the customer question about the stock {ticket} and search the customer wallet CSV file for the stocks.", 
    backstory="""You're the manager of the customer investiments wallet.
    You are the client first contact and you provide the other analystis with the necessary stock ticket and wallet informa""",
    verbose=True,
    llm=llm,
    max_iter=5,
    tools=[csvWalletTool], 
    allow_delegation=False, 
    memory= True
)

yahoo_finance_tool = YahooFinanceTool()

stocketPriceAnalyst = Agent(
    role="Stock Price Analyst",
    goal="Find the {ticket} stock price and analyses price trends. Compare with the price that the customer paid.", 
    backstory="""You're a highly experienced in analyzing the price of specific stocks and
    make predictions about its future price.""",
    verbose=True,
    llm=llm,
    max_iter=5,
    allow_delegation=False,
    memory= True
)

getStockPrice = Task(
    description="Analyze the stock {ticket} price history and create a price trend analyses of up, down or sideways", 
    expected_output="""Specify the current trend stocks price Up, down or sideway. 
    eg. stock= 'AAPL, price UP'.
    """,
    tools=[yahoo_finance_tool],
    agent=stocketPriceAnalyst
)


getCustomerWallet = Task(
    description="""Use the customer question and find the {ticket} in the CSV File.
    Provide if the stock is in the customer wallet and if it is, provide with the mean price he paid
    and the total numbers of stocks onwed.
    """,
    expected_output="If the customer owns the stocks, provide the mean price paid and the total stock numbers.",
    agent = customerManager
)

newsAnalyst = Agent(
    role="News Analyst",
    goal="""Create a short summary of the market news related to the stock {ticket} company.
    Provide a market Fear & Greed Index score about the company.
    For each requested stock asset, specify a number between 0 and 100, where is extreme fear and 100 is extreme greed.""",
    backstory="""You're highly experienced in analyzing market trends and news for more then 10 years.
    You're also a master level analyst in the human psychology.
    You understand the news, their title and information, but you look at those with a health dose of skeptcism.
    You consider the source of the news articles.
    """,
    verbose=True,
    llm=llm,
    max_iter=5,
    allow_delegation=False,
    memory= True
)

searchTool = DuckDuckGoSearchResults(backend='news', num_results=10)

getNews = Task(
    description= f"""Use the search tool to search news about the stock {{ticket}}".
    The current date is {datetime.now()}
    Compose the results into a helpfull report.
    """,
    expected_output="""A summary of the overall market and one paragraph summary for the requested asset. 
    Include the fear/greed score based on the news. Use format:
    <STOCK TICKET>
    <SUMMARY BASED ON NEWS>
    <FEAR/GREED SCORE>
    """,
    agent=newsAnalyst,
    tool=[searchTool]
)

stockRecommender = Agent (
    role="Chief Stock Analyst",
    goal="""Get the data from the customer currently stocks, the provided input of stock price trends and
    the stock news to provide a recommendation: Buy, Sell or Hold the stock.""",
    backstory="""You're the leader of the stock analyst team. You have a great performance in the past 20 years in stock n 
    With all your team informations, you are able to provide the best recommendation for the customer to achieve
    the maximum value creation.""",
    verbose=True,
    llm=llm,
    max_iter=5,
    allow_delegation=True,
    memory= True
)

recommendStock = Task(
    description= """Use the stock price trend, the stock news report and the customers stock mean price of the {ticket} 
    to provide a recommendation: Buy, Sell or Hold. If the previous reports are not well provided you can delegate back 
    to the specific analyst to work again in the their task.""",
    expected_output="""A brief paragraph with the summary of the reasons for recommendation and the r
    ecommendation it self in one of the three possible outputs: Buy, Sell or Hold. Use the format: 
    <SUMMARY OF REASONS>
    <RECOMMENDATION>""",
    agent=stockRecommender,
    context = [getCustomerWallet, getStockPrice, getNews]
)

copyWriter = Agent(
    role="Stock Content Writer",
    goal="""Write an insghtfull compelling and informative 6 paragraph long newletter 
    based on the stock price report, the news report and the recommendation report.""", 
    backstory="""You are a unbeliveble copy writer that understand complex financel concepts 
    and explain for a dummie audience.
    You create complelling stories and narratives that resonate with the audience.""",
    verbose=True,
    llm=llm, 
    max_iter=5,
    allow_delegation=False, 
    memory= True
)


writeNewsletter = Task(
    description="""Use the stock price trend, the stock news report and the stock recommendation to write an insghtfull com 
    Focus on the stock price trend, the news, the fear/greed score and the summary reason for the recommendation. 
    Include the recommendation in the newsletter.""",
    expected_output="""An eloquent 6 paragraph newsletter formated as Markdown in an easy readable manner. It should contai
    - Introduction set the overal picture
    - Main part - provides the meat of the analysis including stock price trend, the news, the fear/greed score and the sum.
    - 3 bullets of the main summary reason of the recommendation
    - Recommendation Summary
    - Recommendation it self""",
    agent=copyWriter,
    context = [getStockPrice, getNews, recommendStock]
)

crew = Crew(
    agents=[customerManager, stocketPriceAnalyst, newsAnalyst, stockRecommender, copyWriter],
    tasks=[getStockPrice, getCustomerWallet, getNews, recommendStock, writeNewsletter],
    verbose=True,
    process=Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)

result = crew.kickoff(inputs={"ticket": "Google Stocks"})