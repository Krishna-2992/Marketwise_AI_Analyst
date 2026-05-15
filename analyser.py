import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

# 1. Initialize the Model (Notice the specific partner package)
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. Define the Prompt
prompt = ChatPromptTemplate.from_template(
    "You are a financial researcher. Summarize the following web search results "
    "about {company} into a professional brief: {results}"
)

# 3. Initialize the Tool
search = TavilySearchResults(max_results=2)

# 4. The LCEL Chain
# Note: We use a Lambda to extract the content from the search tool
chain = (
    {"company": lambda x: x["company"], "results": lambda x: search.invoke(x["company"])}
    | prompt 
    | model 
    | StrOutputParser()
)

result = chain.invoke({"company": "Microsoft"})
print(result)