from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, END
from langchain_core.messages import HumanMessage
import os
from openai import AsyncOpenAI
from app.core.config import settings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.documents import Document
from app.tools.get_bilibili import bilibili_detail_pipiline
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        input: question
        generation: LLM generation
        documents: list of documents
    """

    input: str
    generation: str
    documents: list


class BiliNodes:
    def __init__(self, llm):
        self.llm = llm
        self.retriever = DocumentLoader()
        self.retrieval_grader = self.create_retrieval_grader()

        self.question_rewriter = self.create_question_rewriter()
        self.generate_chain = self.create_generate_chain()

    # BiliNode 1
    async def retrieve(self, state: GraphState):
        """
        根据输入问题检索文档，并将它们添加到图状态中。
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        # print("---节点：开始检索---")
        question = state["input"]
        # print("查询API的问题question:", question)

        # 执行检索
        documents = await self.retriever.get_retriever(keywords=[question], page=1)
        # print(f"这是检索到的Docs:{documents}")
        return {"input": question, "documents": documents}

    # BiliNode 2
    def generate(self, state: GraphState):
        """
        使用输入问题和检索到的文档生成答案，并将生成添加到图形状态中。
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        # print("---节点：生成响应---")

        question = state["input"]
        documents = state["documents"]

        # 基于RAG生成
        generation = self.generate_chain.invoke({"context": documents, "input": question})
        # print(f"生成的响应为:{generation}")

        return {"documents": documents, "input": question, "generation": generation}

    # BiliNode 3
    def grade_documents(self, state: GraphState):
        """
        重新表述输入问题以提高其清晰度和相关性，并使用转换后的问题更新图状态。
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """
        # print("---节点：检查检索到的文档是否与问题相关---")
        question = state["input"]
        documents = state["documents"]

        filtered_docs = []

        for d in documents:
            score = self.retrieval_grader.invoke({"input": question, "document": d.page_content})
            grade = score["score"]
            if grade == "yes":
                # print("---评估结果: 检索文档与问题相关---")
                filtered_docs.append(d)
            else:
                # print("---评估结果: 检索文档与问题不相关---")
                continue

        return {"documents": filtered_docs, "input": question}

    # BiliNode 4
    def transform_query(self, state: GraphState):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        # print("---节点：重写用户输入的问题---")

        question = state["input"]
        documents = state["documents"]

        # 问题重写
        better_question = self.question_rewriter.invoke({"input": question})
        # print(f"这是重写的问题:{better_question}")
        return {"documents": documents, "input": better_question}

    # Helper Function 1
    def create_generate_chain(self):
        """
        Creates a generate chain for answering bilibili-related questions.

        Returns:
            A callable function that takes a context and a question as input and returns a string response.
        """
        generate_template = """
        您是一个人工智能个人助手。用户会提出与BiliBili网站数据相关的问题，这些问题显示在<question></question>标签中。BiliBili网站数据显示在<context></context>标签中。\n

        请根据这些信息组织您的回答。如果用户的问题需要BiliBili API的数据，您可以执行相应的操作。如果找不到答案，请如实回答不知道，不要编造答案。\n

        回答时请使用Markdown格式的字符串。\n
        请注意，除专有名词和学术术语外，所有回答都应使用中文。\n

        回答时请注意以下几点：\n
        - 使用Markdown格式组织内容\n
        - 除专有名词和学术术语外，全部使用中文\n
        - 输出必须涵盖<context></context>标签中与查询相关的所有信息，并明确引用上下文信息\n
        - 如果无法提供答案，请回复"抱歉，我无法回答这个问题。"\n
        - 如果问题与查询结果无关，请在回答中说明\n

        <context>
        {context}
        </context>

        <question>
        {input}
        </question>
        """

        generate_prompt = PromptTemplate(template=generate_template, input_variables=["context", "input"])

        # Create the generate chain
        generate_chain = generate_prompt | self.llm | StrOutputParser()

        return generate_chain

    # Helper Function 2
    # 创建评估检索文档与用户问题相关性的评分器
    def create_retrieval_grader(self):
        """
        Creates a retrieval grader that assesses the relevance of a retrieved document to a user question.

        Returns:
            A callable function that takes a document and a question as input and returns a JSON object with a binary score indicating whether the document is relevant to the question.
        """

        # 使用的特殊标记是为了指定不同部分的开始和结束，以及明确不同类型的文本块。
        # 这些标记可以帮助大模型更好地理解和区分输入数据的不同部分，从而更精确地执行特定的任务。
        # 您是一名评分员，负责评估检索到的文档与用户问题的相关性。如果文档包含与用户问题相关的关键词，请将其评为相关。这不需要非常严格的测试。目标是过滤掉错误的检索结果。
        grade_prompt = PromptTemplate(
            template="""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            您是一名评估员，负责评估检索到的文档与用户问题的相关性。如果文档包含与用户问题相关的关键词，请将其评为相关。不需要非常严格的测试，目标是过滤掉错误的检索结果。
            请给出"yes"或"no"的二元评分来指示文档是否与问题相关。
            提供一个包含单个键'score'的JSON，不要有前言或解释。
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>

            检索到的文档：\n\n {document} \n\n
            用户问题：{input} \n
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["document", "input"],
        )

        # 创建一个 检索 的链
        retriever_grader = grade_prompt | self.llm | JsonOutputParser()

        return retriever_grader

    # Helper Function 3
    # 您是一个问题重写器，将输入的问题转换成更好的版本，优化以适应向量存储检索。请查看输入并尝试理解其潜在的语义意图/含义。
    def create_question_rewriter(self):
        """
        Creates a question rewriter chain that rewrites a given question to improve its clarity and relevance.

        Returns:
            A callable function that takes a question as input and returns the rewritten question as a string.
        """
        re_write_prompt = PromptTemplate(
            template="""
            您是一个问题重写器，将输入的问题转换成更好的版本以优化向量存储检索。请分析输入并理解其潜在的语义意图/含义。

            初始问题：{input}

            请生成一个优化后的改进问题。""",

            input_variables=["input"],
        )

        question_rewriter = re_write_prompt | self.llm | StrOutputParser()

        return question_rewriter


# 由于Bilibili网站的检索和Embedding较复杂, 将该过程定义为一个类
class DocumentLoader:
    """
    This class uses the get_docs function to take a Keyword as input, and outputs a list of documents (including metadata).
    """

    async def get_docs(self, keywords: List[str], page: int) -> List[Document]:
        """
        Asynchronously retrieves documents based on specific keywords from the BiliBili API.
        This function utilizes a pipeline to fetch and format video data, returning it as Document objects.

        Args:
        keywords (List[str]): A list of keywords used to query the BiliBili API.
        page (int): The page number in the API request, used for pagination.

        Returns:
            List[Document]: A list of Document objects containing the retrieved content.
        """

        raw_docs = await bilibili_detail_pipiline(keywords=keywords, page=page)

        docs = [Document(page_content=doc["real_data"]) for doc in raw_docs]

        return docs

    async def create_vector_store(self, docs: List[Document], store_path: Optional[str] = None) -> 'FAISS':
        """
        Creates a FAISS vector store from a list of documents.

        Args:
            docs (List[Document]): A list of Document objects containing the content to be stored.
            store_path (Optional[str]): The path to store the vector store locally. If None, the vector store will not be stored.

        Returns:
            FAISS: The FAISS vector store containing the documents.
        """
        # 执行文本切分，并使用OpenAI Embedding模型生成向量表示
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        texts = text_splitter.split_documents(docs)
        embedding_model = OpenAIEmbeddings()
        store = FAISS.from_documents(texts, embedding_model)

        if store_path:
            store.save_local(store_path)
        return store

    async def get_retriever(self, keywords: List[str], page: int):
        """
        Retrieves documents and returns a retriever based on the documents.

        Args:
            keywords (List[str]): Keywords to search documents.
            page (int): Page number for pagination of results.

        Returns:
            Retriever instance or FAISS vector store.
        """
        # print(f"开始实时查询BiliBiliAPI获取数据")
        docs = await self.get_docs(keywords, page)
        # print(f"接收到的BiliBili数据为：{docs}")
        # print("-------------------------")
        # print(f"开始进行向量数据库存储")
        vector_store = await self.create_vector_store(docs)
        # print(f"成功完成向量数据库的存储")
        # print("-------------------------")
        # print(f"开始进行文本检索")
        retriever = vector_store.as_retriever(search_kwargs={"k": 6})
        retriever_result = retriever.invoke(str(keywords))
        # print(f"检索到的数据为：{retriever_result}")
        # print(type(retriever_result))
        return retriever_result


class BiliEdge:
    def __init__(self, llm):
        self.llm = llm
        self.hallucination_grader = self.create_hallucination_grader()
        self.content_evaluator = self.create_content_evaluator()

    def decide_to_generate(self, state):
        """
        根据过滤后的文档与输入问题的相关性确定是生成答案还是重新生成问题。如果所有文档都不相关，则决定转换查询；否则，它决定生成答案。
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        # print("---进入检索文档与问题相关性判断---")

        filtered_documents = state["documents"]

        # 判断filtered_documents是否为空, 确定下一步节点
        if not filtered_documents:
            # print("---决策：所有检索到的文档均与问题无关，转换查询---")
            return "transform_query"
        else:
            # print("---决策：生成最终响应---")
            return "generate"

    def grade_generation_v_documents_and_question(self, state):
        """
        根据文档的基础及其解决问题的能力来评估生成的答案。如果基于既定事实解决了问题，那么它被认为是有用的；否则，它不受支持或无用。
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """
        # print("---检查是否输入模型幻觉输出---")
        question = state["input"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score["score"]

        if grade == "yes":
            # print("---决策: 生成内容是基于检索到的文档的既定事实---")

            # print("---检查最终响应是否与输入的问题相关---")
            score = self.content_evaluator.invoke({"input": question, "generation": generation, "documents": documents})
            grade = score["score"]
            if grade == "yes":
                # print("---判定: 生成响应与输入问题相关---")
                return "useful"
            else:
                # print("---判定: 生成响应与输入问题不相关---")
                return "not useful"
        else:
            # print("---判定：生成响应与检索文档不相关，模型进入幻觉状态---")
            return "not supported"

    # Helper Function 1
    # 您是一名评分员，负责评估答案是否基于/得到一组事实的支持。请给出“是”或“否”的二元评分，以表明答案是否基于/得到事实的支持。提供一个只有一个键“score”的JSON，不需要前言或解释。
    def create_hallucination_grader(self):
        """
        Creates a hallucination grader that assesses whether an answer is grounded in/supported by a set of facts.

        Returns:
            A callable function that takes a generation (answer) and a list of documents (facts) as input and returns a JSON object with a binary score indicating whether the answer is grounded in/supported by the facts.
        """
        hallucination_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            您是一名评分员，负责评估答案是否基于给定的事实依据。请给出"yes"或"no"的二元评分来指示答案是否有事实支持。
            提供一个包含单个键'score'的JSON，不要有前言或解释。
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            事实依据：
            \n ------- \n
            {documents}
            \n ------- \n
            待评估答案：{generation}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "documents"],
        )

        hallucination_grader = hallucination_prompt | self.llm | JsonOutputParser()

        return hallucination_grader

    # Helper Function 2
    def create_content_evaluator(self):
        """
        创建一个内容评估器，用于评估生成的内容是否与给定问题相关。

        该评估器将生成的内容、问题和相关文档作为输入，返回一个包含评估结果的JSON对象。评估结果包括一个二进制评分（"yes" 或 "no"）以及简要的反馈说明，评估内容是否与问题相关，或者需要哪些改进。

        返回：
            一个可调用的函数，该函数接受生成的内容（generation）、问题（input）和相关文档（documents）作为输入，返回一个包含以下内容的JSON对象：
                - "score": 二进制评分（"yes" 或 "no"），表示生成的内容是否与问题相关。
                - "feedback": 简要的反馈说明，包括评估过程中发现的相关性问题或改进建议。

        示例：
            输入：生成的内容、问题和相关文档。
            输出：一个JSON对象，包含评分和反馈说明。
        """
        eval_template = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
            您是一名内容评估员，负责判断生成内容是否与给定问题相关。
            请提供包含以下键的JSON响应：

            "score": 二元评分"yes"或"no"，指示内容是否与问题相关
            "feedback": 简要的评估说明，包括内容相关性存在的问题或改进建议

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            生成内容：
            \n ------- \n
            {generation}
            \n ------- \n
            原始问题：{input}
            \n ------- \n
            相关文档：{documents}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "input", "documents"],
        )

        content_evaluator = eval_template | self.llm | JsonOutputParser()

        return content_evaluator


class BiliSearchTool:
    def __init__(self):
        self.client = ChatOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            model=settings.DEEPSEEK_MODEL,
            temperature=0
        )
        self.model = settings.DEEPSEEK_MODEL
        self.chain = self.create_workflow()

    def create_workflow(self):
        """
        创建并初始化工作流以及其组成的节点和边。

        Returns:
        StateGraph: 完全初始化和编译好的工作流对象。
        """
        # 初始化图结构
        workflow = StateGraph(GraphState)

        # 创建图节点的实例
        bili_nodes = BiliNodes(llm=self.client)

        # 创建边节点的实例
        edge_graph = BiliEdge(llm=self.client)

        # 定义节点
        workflow.add_node("bili_analysis", bili_nodes.retrieve)  # retrieve documents
        workflow.add_node("grade_documents", bili_nodes.grade_documents)  # grade documents
        workflow.add_node("generate", bili_nodes.generate)  # generate answers
        workflow.add_node("transform_query", bili_nodes.transform_query)  # transform query

        # 创建图
        workflow.set_entry_point("bili_analysis")
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
                "useful": END,
                "not useful": "transform_query",
            }
        )

        # 编译图
        chain = workflow.compile()

        return chain
    

if __name__ == '__main__':
    tool = BiliSearchTool()
    chain = tool.create_workflow()

    async def test():
        # 这个 thread_id 可以取任意数值
        config = {"configurable": {"thread_id": "1"}}

        input_text = "你好，我叫木羽"
        input_all = {"input": input_text, "generation": "NULL", "documents": "NULL"}
        async for chunk in chain.astream(input_all, config, stream_mode="values"):
            print(chunk)

    import asyncio
    asyncio.run(test())
