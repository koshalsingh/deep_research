from agents import Agent, WebSearchTool, ModelSettings
from agents import Agent, OpenAIChatCompletionsModel, ModelSettings, Runner, function_tool
from openai import AsyncOpenAI
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import os

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

@function_tool
def web_search(query: str) -> str:
    """Searches the web for up-to-date information on a given query."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
        return "\n".join([r["body"] for r in results])

# 3. Define instructions and task
INSTRUCTIONS = """
You are a research assistant. Given a search term, you search the web for that term and
produce a concise summary of the results. The summary must be 2-3 paragraphs and less than 300 words.
Capture the main points and be succinct. Reply only with the summary.
"""

task = "Most popular AI Agent frameworks in 2026"

# 4. Create your Agent with the function_tool
settings = ModelSettings(tool_choice="required")

search_agent = Agent(
    name="Search Agent",
    instructions=INSTRUCTIONS,
    tools=[web_search],  # Pass the function_tool here
    model=MODEL_NAME,
    model_settings=settings)