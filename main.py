import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START ,END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

class MessageState(MessagesState):
    pass

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model_name = "gemini-2.5-flash"

llm = ChatGoogleGenerativeAI(model=model_name , google_api_key=GOOGLE_API_KEY)

def multiply(a: int, b: int):
    """
    multiply a and b
    
    args:
        a: first int
        b:second int
    """
    return  f" {a} * {b} =  {a*b}"

llm_with_tools = llm.bind_tools([multiply])
# tool_call = llm_with_tools.invoke([HumanMessage(content="multiply 2 and 3",name="vijay")])

def tool_calling_llm(state: MessagesState):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)

builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode([multiply]))
builder.add_edge(START,"tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition
)
builder.add_edge("tool_calling_llm",END)

graph = builder.compile()

messages = graph.invoke({"messages":[HumanMessage(content="write a poem about nature",name="Vijay")]})

for m in messages["messages"]:
    m.pretty_print()