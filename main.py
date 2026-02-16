from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
# from langchain_community.tools import Tool
# from langchain_google_community import GoogleSearchAPIWrapper
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9, api_key=GOOGLE_API_KEY)
agent = create_agent(tools=[], model=model, debug=True)
result = agent.invoke({
    "messages": [HumanMessage(content="What is the capital of France?")]
})

print(result)

