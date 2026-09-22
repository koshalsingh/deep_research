from pydantic import BaseModel, Field
# from agents import Agent
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from duckduckgo_search import DDGS
from agents import Agent, OpenAIChatCompletionsModel, ModelSettings, Runner, function_tool
from IPython.display import display, Markdown

load_dotenv(override=True)

external_client = AsyncOpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "gpt-5.4-mini")
MODEL_NAME = OpenAIChatCompletionsModel(
    model="gemini-3.1-flash-lite", 
    openai_client=external_client
)
HOW_MANY_SEARCHES = int(os.getenv("HOW_MANY_SEARCHES", 5))


INSTRUCTIONS = f"""
You are a research assistant. Given a user query, come up with a set of web searches
to perform to best answer the query. Output {HOW_MANY_SEARCHES} terms to query for.
"""

class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")
    
planner_agent = Agent(name="Planner Agent", instructions=INSTRUCTIONS, model=MODEL_NAME, output_type=WebSearchPlan)