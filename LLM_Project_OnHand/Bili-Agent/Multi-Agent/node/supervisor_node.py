from config.graph import GraphState
from config.Router import Router

from langgraph.graph import END
from langchain_core.messages import AIMessage, SystemMessage

members = ["chat", "bili_analysis", "arxiv_retriever"]
options = members + ["FINISH"]


# 主Agent, 进行决策和选择字Agent
class SupervisorNode:
    def __init__(self, llm):
        self.llm = llm

    def supervisor(self, state: GraphState):
        system_prompt = (
            f"You are a supervisor tasked with managing and coordinating a conversation between the following workers: {members}.\n\n"
    
            "Each worker has a specific role:\n"
            """
            - chat: Responds to user inputs using natural language. This worker handles general conversation, provides explanations, and clarifies user queries.\n
            - bili_analysis: A video search node for the Bilibili website. If the user's question mentions keywords such as 'Bilibili', 'video', or refers to specific video content, this worker will handle the task.\n
            - arxiv_retriever: A research paper analysis node for the arXiv website. If the user's question refers to research papers, academic topics, or mentions 'arXiv', this worker will be responsible for fetching and analyzing the relevant papers.\n
            """
    
            """
            As the supervisor, your job is to determine which worker is best suited to respond based on the user's question. \n
    
            Your response should ONLY include the name of the worker from the following list: 'chat', 'bili_analysis', 'arxiv_retriever'. Do not provide any other information. Once the worker completes their task, they will provide results, and you will continue coordinating.\n
    
            Once all tasks are completed and you have received the final responses from all relevant workers, respond with 'FINISH'. If the last message is from 'AIMessage', also respond with 'FINISH' to signal the end of the process.\n
    
            Your sole responsibility is to route the task to the appropriate worker and respond accordingly, only using the worker names.\n
            """
        )
        messages = state["messages"]
        input = state["input"]

        message = [SystemMessage(content=system_prompt), ] + messages

        if state["messages"] != [] and isinstance(state["messages"][-1], AIMessage):
            return {"next": END}

        response = self.llm.with_structured_output(Router).invoke(input=message)

        # 判断llm输出的结果, 结构化输出, 由自己定义情况
        next_ = response["next"]

        if next_ == "FINISH":
            next_ = END

        return {"input": input, "generation": state["generation"], "documents": state["documents"], "next": next_}
