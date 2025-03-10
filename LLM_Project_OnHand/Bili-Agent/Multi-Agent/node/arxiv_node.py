from langchain_core.messages import AIMessage

from langchain_core.documents import Document
from typing import List
from tools import get_arxiv
from config.graph import GraphState
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class ArxivNodes:
    def __init__(self, llm):
        self.llm = llm

    # ArxivNode 1
    def Arxiv_retrieve(self, state: GraphState):
        """
        根据输入问题检索文档，并将它们添加到图状态中。
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---节点：开始检索---")
        question = state["input"]
        print("查询API的问题question:", question)

        # 执行检索
        documents = self.get_arxiv_retriever(query=question, top_k_results=10)
        print(f"这是检索到的Docs:{documents}")
        return {"input": question, "documents": documents}

    # ArxivNode 2
    def Arxiv_generate(self, state: GraphState):
        """
        使用输入问题和检索到的文档生成答案，并将生成添加到图形状态中。
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---节点：生成响应---")

        question = state["input"]
        documents = state["documents"]

        chain = self.create_generate_chain()
        generation = chain.invoke({"context": documents, "input": question})
        print(f"生成的响应为:{generation}")
        final_response = [AIMessage(content=generation, name="arxiv")]  # 这里要添加名称

        return {"documents": documents, "input": question, "generation": generation, "messages": final_response, "next": "NULL"}

    # Helper function
    def get_arxiv_retriever(self, query: str, top_k_results: int) -> List[Document]:
        """
        Retrieves documents and returns a retriever based on the documents.

        Args:
            query (str): Keywords to search documents.
            top_k_results (int): Paper number of results.

        Returns:
            Retriever instance.
        """
        arxiv_search = get_arxiv.ArxivAPIWrapper()
        arxiv_search.top_k_results = top_k_results

        print(f"开始实时查询Arxiv-API获取数据")
        docs = arxiv_search.get_summaries_as_docs(query)
        print(f"接收到的Arxiv数据为：{docs}")
        print("-------------------------")

        return docs

    # Helper function
    def create_generate_chain(self):
        """
        Creates a generate chain for answering arxiv-related questions.

        Returns:
            A callable function that takes a context and a question as input and returns a string response.
        """
        generate_template = """
            You are an AI personal assistant. The user will ask questions related to data from the Arxiv website, with the user's question enclosed in the <question></question> tags. The results of the query will be displayed in the <context></context> tags.\n

            Please organize your response based on this information. If the user's question requires data from the Arxiv API, you may perform the corresponding operation. If an answer cannot be found, please answer honestly that you do not know, and do not fabricate an answer.\n

            For your answer, please answer in the format of a string composed of Markdown. \n
            Please note that all your answers should be in Chinese, except for proprietary and academic terms.\n

            When responding, please keep in mind the following points:\n
            - Organize content in Markdown format.\n
            - All your answers should be in Chinese, except for proprietary and academic terms.\n
            - The output must cover all relevant information from the query result within the <context></context> tags, and clearly reference the context information.\n
            - If an answer cannot be provided, reply with "Sorry, I cannot answer this question."\n
            - If the question is unrelated to the query results, explain that in your response.\n


            <context>
            {context}
            </context>

            <question>
            {input}
            </question>
        """

        # """
        #     你是一个人工智能个人助手。用户会提出与Arxiv论文网站数据相关的问题, 其中, 用户问题会显示在<question></question>标签中, 基于问题查询的结果会显示在<context></context>标签中。
        #
        #     请根据这些信息来组织你的回答。如果用户的问题需要通过Arxiv API获取数据，你可以进行相应的操作。如果无法找到答案，请诚实地回答你不知道，而不要编造答案。
        #
        #     请确保所有的回答以Markdown格式呈现，且使用“```markdown”作为回答的开头。
        #
        #     在回答时，请注意以下几点：
        #     - 以Markdown格式组织内容。
        #     - 输出的答案必须覆盖查询结果<context></context>标签中的所有重要信息，并清晰地引用上下文信息。
        #     - 当无法回答时，请用“抱歉，我无法回答这个问题”。
        #     - 如果问题与查询结果无关，答复时请说明这一点。
        #
        #     示例：
        #           这是一个Markdown格式的回答示例。
        #           根据搜索到的上下文信息，我得出结论...
        #
        #     <context>
        #     {context}
        #     </context>
        #
        #     <question>
        #     {input}
        #     </question>
        # """

        generate_prompt = PromptTemplate(template=generate_template, input_variables=["context", "input"])

        # Create the generate chain
        generate_chain = generate_prompt | self.llm | StrOutputParser()

        return generate_chain
