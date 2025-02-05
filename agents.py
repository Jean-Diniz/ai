"""
GRUPO DE AGENTES PARA ESCREVER ARTIGOS
    1. Planner - Planejador do artigo
    2. Writer - Escritor do artigo
    3. Editor - Editor do artigo
"""


from crewai import Agent, Task, Crew
import os
from dotenv import load_dotenv
from IPython.display import Markdown

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')


planner = Agent(
    role = "Content Planner",
    goal = "Plan engaging and factually accurate content about the {topic}.",
    backstory = """You're working on planning a blog post article about the topic: {topic}.
    You collect information that helps the audience learn something and make informed decisions.
    Your work is the basis for the Content Writer to write the arcticle on this topic.
    """,
    allow_delegation=False,
    verbose=True,
    llm="groq/llama3-8b-8192"
)

writer = Agent(
    role = "Content Writer",
    goal = "Write insightfull and factually accurate opinion piece about the topic: {topic}.",
    backstory = """You're working on writing an opinion about the topic: {topic}.
    You base your opinion on the work of the Content Planner, who provides an outline and relevant context about the topic. 
    You follow the main objectives and direction of the outline, as provided by the Content Planner. 
    You also provide objective and impartial insights and back them with information provided by the Content Planner.
    """,
    allow_delegation=False,
    verbose=True,
    llm="groq/llama3-8b-8192"
)

editor = Agent(
    role = "Editor",
    goal = "Edit a given blog post article to align with the writing style of the organization.",
    backstory = """You are a editor that receives a blog post from the Content Writer.
    Your goal is to review the blog post to ensure that it follows journalistic best practices.
    """,
    allow_delegation=False,
    verbose=True,
    llm="groq/llama3-8b-8192"
)

plan = Task(
    description = (
        "1. Prioritaze the last trends, key players, and noteworthy news on {topic}."
        "2. Identify the target audience, considering their interests and pain points."
        "3. Develop a detailed content outline including and introduction, key points, and call to action."
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output = "A comprehensive content plan document with an outline, audience analysis, SEO keywords and resources.",
    agent = planner
)

write = Task(
    description = (
        "1. Use the content planto to craft a compelling blog post on topic: {topic}."
        "2. Incorporate the SEO keywords naturally."
        "3. Section/Subtitles are properly named in engaging manner."
        "4. Ensure the post is structured with engaging introduction, insightful body, and summarization conclusion."
    ),
    expected_output = "A well-written blog post in Markdown format, ready for publication, each section should have 2 or 3 peragraphs.",
    agent = writer
)

edit = Task(
    description = ("Proofread the given blog post for grammatical errrors and aligment with tha brand's voice."),
    expected_output = "A well-written blog post in Markdown format, ready for publication, each section should have 2 or 3 peragraphs.",
    agent = editor
)

crew = Crew(
    agents = [planner, writer, editor],
    tasks = [plan, write, edit]
)

result = crew.kickoff(inputs={"topic": "Artificial Intelligence"})
Markdown(result.raw)
