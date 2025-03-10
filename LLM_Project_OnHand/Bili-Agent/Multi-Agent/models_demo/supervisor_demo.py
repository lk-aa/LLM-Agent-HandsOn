import getpass
import os
from langchain_openai import ChatOpenAI


# 如果用开源模型，可以用Ollama 接入
# from langchain_ollama import ChatOllama

# llm = ChatOllama(
#     base_url = "http://192.168.110.131:11434",  # 注意：这里需要替换成自己本地启动的endpoint
#     model="qwen2.5:72b",
# )


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

llm = ChatOpenAI(model="gpt-3.5-turbo")

from langgraph.graph import StateGraph, MessagesState, START, END


class AgentState(MessagesState):
    next: str


members = ["chat", "coder", "sqler"]
options = members + ["FINISH"]


from typing import Literal
from typing_extensions import TypedDict


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH"""
    next: Literal[*options]


from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage


def supervisor(state: AgentState):
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}.\n\n"
        "Each worker has a specific role:\n"
        "- chat: Responds directly to user inputs using natural language.\n"
        "- coder: Activated for tasks that require mathematical calculations or specific coding needs.\n"
        "- sqler: Used when database queries or explicit SQL generation is needed.\n\n"
        "Given the following user request, respond with the worker to act next."
        " Each worker will perform a task and respond with their results and status."
        " When finished, respond with FINISH."
    )

    messages = [{"role": "system", "content": system_prompt}, ] + state["messages"]

    response = llm.with_structured_output(Router).invoke(messages)

    next_ = response["next"]

    if next_ == "FINISH":
        next_ = END

    return {"next": next_}


def chat(state: AgentState):
    messages = state["messages"][-1]
    model_response = llm.invoke(messages.content)
    final_response = [HumanMessage(content=model_response.content, name="chat")]   # 这里要添加名称
    return {"messages": final_response}


def coder(state: AgentState):
    messages = state["messages"][-1]
    model_response = llm.invoke(messages.content)
    final_response = [HumanMessage(content=model_response.content, name="coder")]   # 这里要添加名称
    return {"messages": final_response}


def sqler(state: AgentState):
    messages = state["messages"][-1]
    model_response = llm.invoke(messages.content)
    final_response = [HumanMessage(content=model_response.content, name="sqler")]  # 这里要添加名称
    return {"messages": final_response}


builder = StateGraph(AgentState)

# builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor)
builder.add_node("chat", chat)
builder.add_node("coder", coder)
builder.add_node("sqler", sqler)


for member in members:
    # 我们希望我们的工人在完成工作后总是向主管“汇报”
    builder.add_edge(member, "supervisor")


builder.add_conditional_edges("supervisor", lambda state: state["next"])

# 添加开始和节点
builder.add_edge(START, "supervisor")

# 编译图
graph = builder.compile()


# from IPython.display import Image, display

# display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

for chunk in graph.stream({"messages": "你好，请你介绍一下你自己"}, stream_mode="values"):
    print(chunk)

for chunk in graph.stream({"messages": "你好，帮我生成一个二分查找的Python代码"}, stream_mode="values"):
    print(chunk)

for chunk in graph.stream({"messages": "我想查询数据库中 data 表的所有数据，"}, stream_mode="values"):
    print(chunk)

all_chunk = []

for chunk in graph.stream({"messages": "我想查询数据库中 data 表的所有数据，"}, stream_mode="values"):
    all_chunk.append(chunk)
