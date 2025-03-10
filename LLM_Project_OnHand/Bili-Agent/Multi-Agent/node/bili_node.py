from langchain_core.messages import AIMessage

from config.graph import GraphState
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.documents import Document
from tools import get_bilibili
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


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
        print("---节点：开始检索---")
        question = state["input"]
        print("查询API的问题question:", question)

        # 执行检索
        documents = await self.retriever.get_retriever(keywords=[question], page=1)
        print(f"这是检索到的Docs:{documents}")
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
        print("---节点：生成响应---")

        question = state["input"]
        documents = state["documents"]

        # 基于RAG生成
        generation = self.generate_chain.invoke({"context": documents, "input": question})
        print(f"生成的响应为:{generation}")
        final_response = [AIMessage(content=generation, name="chat")]  # 这里要添加名称

        return {"documents": documents, "input": question, "generation": generation, "messages": final_response, "next": "NULL"}

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
        print("---节点：检查检索到的文档是否与问题相关---")
        question = state["input"]
        documents = state["documents"]

        filtered_docs = []

        for d in documents:
            score = self.retrieval_grader.invoke({"input": question, "document": d.page_content})
            grade = score["score"]
            if grade == "yes":
                print("---评估结果: 检索文档与问题相关---")
                filtered_docs.append(d)
            else:
                print("---评估结果: 检索文档与问题不相关---")
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
        print("---节点：重写用户输入的问题---")

        question = state["input"]
        documents = state["documents"]

        # 问题重写
        better_question = self.question_rewriter.invoke({"input": question})
        print(f"这是重写的问题:{better_question}")
        return {"documents": documents, "input": better_question}

    # Helper Function 1
    def create_generate_chain(self):
        """
        Creates a generate chain for answering bilibili-related questions.

        Returns:
            A callable function that takes a context and a question as input and returns a string response.
        """
        generate_template = """
        You are an artificial intelligence personal assistant. Users will raise questions related to BiliBili website data, which are displayed in the section attached to the<question></question>tag. The BiliBili website data is displayed in the section attached to the<context></context>tag.\n

        Please organize your response based on this information. If the user's question requires data from the BiliBili API, you may perform the corresponding operation. If an answer cannot be found, please answer honestly that you do not know, and do not fabricate an answer.\n

        For your answer, please answer in the format of a string composed of Markdown. \n
        Please note that all your answers should be in Chinese, except for proprietary and academic terms.\n

        When responding, please keep in mind the following points:\n
        - Organize content in Markdown format.\n
        - All your answers should be in Chinese, except for proprietary and academic terms.
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
            You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>

            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {input} \n
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
            You a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval. Look at the input and reference to reason about the underlying sematic intent / meaning.

            Here is the initial question: {input}

            Formulate an improved question.""",

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

        raw_docs = await get_bilibili.bilibili_detail_pipiline(keywords=keywords, page=page)

        docs = [Document(page_content=doc["real_data"]) for doc in raw_docs]

        return docs

    async def create_vector_store(self, docs, store_path: Optional[str] = None) -> 'FAISS':
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
        print(f"开始实时查询BiliBiliAPI获取数据")
        docs = await self.get_docs(keywords, page)
        print(f"接收到的BiliBili数据为：{docs}")
        print("-------------------------")
        print(f"开始进行向量数据库存储")
        vector_store = await self.create_vector_store(docs)
        print(f"成功完成向量数据库的存储")
        print("-------------------------")
        print(f"开始进行文本检索")
        retriever = vector_store.as_retriever(search_kwargs={"k": 6})
        retriever_result = retriever.invoke(str(keywords))
        print(f"检索到的数据为：{retriever_result}")
        print(type(retriever_result))
        return retriever_result
