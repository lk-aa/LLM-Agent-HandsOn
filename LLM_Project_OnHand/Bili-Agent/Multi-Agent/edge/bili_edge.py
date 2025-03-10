from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


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
        print("---进入检索文档与问题相关性判断---")

        filtered_documents = state["documents"]

        # 判断filtered_documents是否为空, 确定下一步节点
        if not filtered_documents:
            print("---决策：所有检索到的文档均与问题无关，转换查询---")
            return "transform_query"
        else:
            print("---决策：生成最终响应---")
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
        print("---检查是否输入模型幻觉输出---")
        question = state["input"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score["score"]

        if grade == "yes":
            print("---决策: 生成内容是基于检索到的文档的既定事实---")

            print("---检查最终响应是否与输入的问题相关---")
            score = self.content_evaluator.invoke({"input": question, "generation": generation, "documents": documents})
            grade = score["score"]
            if grade == "yes":
                print("---判定: 生成响应与输入问题相关---")
                return "useful"
            else:
                print("---判定: 生成响应与输入问题不相关---")
                return "not useful"
        else:
            print("---判定：生成响应与检索文档不相关，模型进入幻觉状态---")
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
            You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Here are the facts:
            \n ------- \n
            {documents}
            \n ------- \n
            Here is the answer: {generation}
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
            You are a content evaluator tasked with determining whether the generated content is relevant to the given question.
            Please provide a JSON response with the following keys:

            "score": A binary score "yes" or "no" indicating whether the content is relevant to the question. Use "..." instead of '...'.
            "feedback": A brief explanation of your evaluation, including any issues with the relevance of the content to the question or areas for improvement.

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the generated content:
            \n ------- \n
            {generation}
            \n ------- \n
            Here is the question: {input}
            \n ------- \n
            Here are the relevant documents: {documents}
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["generation", "input", "documents"],
        )

        content_evaluator = eval_template | self.llm | JsonOutputParser()

        return content_evaluator

