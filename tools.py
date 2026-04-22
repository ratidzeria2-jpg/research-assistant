from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool

search = DuckDuckGoSearchResults()
wiki = WikipediaAPIWrapper()

@tool
def search_tool(query: str) -> str:
    """Search the web for information. Use this when you need to find current information or news."""
    return search.run(query)

@tool
def wiki_tool(query: str) -> str:
    """Search Wikipedia for information. Use this when you need detailed background knowledge on a topic."""
    return wiki.run(query)