from config.graph import GraphState
from langchain_core.messages import AIMessage


class ChatNode:
    def __init__(self, llm):
        self.llm = llm

    def chat(self, state: GraphState):
        input = state["input"]  # 传入全部信息
        messages = state["messages"]
        print("chat:", messages)

        model_response = self.llm.invoke(messages)
        final_response = [AIMessage(content=model_response.content, name="chat")]  # 这里要添加名称
        return {"input": input, "generation": model_response.content, "messages": final_response, "next": "NULL",
                "documents": state["documents"]}
