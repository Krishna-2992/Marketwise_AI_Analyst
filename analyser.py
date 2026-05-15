import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

class CompanyAnalysis(BaseModel):
    summary: str = Field(description="A professional brief summarizing the company's recent events.")
    sentiment_score: int = Field(description="A sentiment score from 1 to 10, where 10 is incredibly bullish.")

# 1. Initialize the Model (Notice the specific partner package)
model = ChatOpenAI(model="gpt-4o", temperature=0)

structured_model = model.with_structured_output(CompanyAnalysis)

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
    | structured_model 
)

result = chain.invoke({"company": "Microsoft"})
print("---")
print("Summary:", result.summary)
print("---")
print("Sentiment Score:", result.sentiment_score)