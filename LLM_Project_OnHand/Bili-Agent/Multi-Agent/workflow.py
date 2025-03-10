from langchain_openai import ChatOpenAI
from config.graph import GraphState
from node import arxiv_node, bili_node, chat_node, supervisor_node
from edge import bili_edge

from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage
# 导入检查点
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def create_workflow(api_key: str, model: str):
    """
    创建并初始化工作流以及其组成的节点和边。

    Returns:
    StateGraph: 完全初始化和编译好的工作流对象。
    """

    # 创建 LLM model 实例，配置为使用 GPT-4o 模型和指定的温度参数
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.chatanywhere.org/v1",
        model=model,
        temperature=0
    )

    # 初始化图结构
    workflow = StateGraph(GraphState)

    # 创建图节点的实例
    supervisor_nodes = supervisor_node.SupervisorNode(llm=llm)
    bili_nodes = bili_node.BiliNodes(llm=llm)
    arxiv_nodes = arxiv_node.ArxivNodes(llm=llm)
    chat_nodes = chat_node.ChatNode(llm=llm)

    # 创建边节点的实例
    edge_graph = bili_edge.BiliEdge(llm=llm)

    # 定义节点
    workflow.add_node("supervisor", supervisor_nodes.supervisor)

    workflow.add_node("chat", chat_nodes.chat)

    workflow.add_node("bili_analysis", bili_nodes.retrieve)  # retrieve documents
    workflow.add_node("grade_documents", bili_nodes.grade_documents)  # grade documents
    workflow.add_node("generate", bili_nodes.generate)  # generate answers
    workflow.add_node("transform_query", bili_nodes.transform_query)  # transform query

    workflow.add_node("arxiv_retriever", arxiv_nodes.Arxiv_retrieve)
    workflow.add_node("arxiv_generate", arxiv_nodes.Arxiv_generate)

    # 创建图
    workflow.add_edge(START, "supervisor")
    # workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges("supervisor",
                                   lambda state: state["next"],
                                   )
    workflow.add_edge("chat", "supervisor")
    workflow.add_edge("bili_analysis", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        edge_graph.decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        }
    )
    workflow.add_edge("transform_query", "bili_analysis")
    workflow.add_conditional_edges(
        "generate",
        edge_graph.grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": "supervisor",
            "not useful": "transform_query",
        }
    )

    workflow.add_edge("arxiv_retriever", "arxiv_generate")
    workflow.add_edge("arxiv_generate", "supervisor")

    # 编译图, 添加聊天历史记录和检查点
    memory = MemorySaver()
    chain = workflow.compile(checkpointer=memory)

    # 绘制图结构
    # chain.get_graph().print_ascii()
    # chain.get_graph().draw_png("supervisor.png")
    # chain.get_graph(xray=True).draw_mermaid_png(output_file_path="supervisor2.png")

    return chain


if __name__ == '__main__':
    chain = create_workflow(
        os.getenv('OPENAI_API_KEY'),
        os.getenv('model')
    )

    async def test():
        # 这个 thread_id 可以取任意数值
        config = {"configurable": {"thread_id": "1"}}

        input_text = "你好，我叫木羽"
        message = [HumanMessage(content=input_text, name="user_chat")]
        input_all = {"input": input_text, "generation": "NULL", "messages": message, "next": "NULL", "documents": "NULL"}
        async for chunk in chain.astream(input_all, config, stream_mode="values"):
            print(chunk)

        input_text = "请推荐几个bilibili上有关 LangGraph 的视频。"
        message = [HumanMessage(content=input_text, name="user_chat")]
        input_all = {"input": input_text, "generation": "NULL", "messages": message, "next": "NULL", "documents": "NULL"}
        async for chunk in chain.astream(input_all, config, stream_mode="values"):
            print(chunk)

        input_text = "请问我叫什么？"
        message = [HumanMessage(content=input_text, name="user_chat")]
        input_all = {"input": input_text, "generation": "NULL", "messages": message, "next": "NULL", "documents": "NULL"}
        async for chunk in chain.astream(input_all, config, stream_mode="values"):
            print(chunk)

    import asyncio
    asyncio.run(test())
