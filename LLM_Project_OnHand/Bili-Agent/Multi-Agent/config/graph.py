from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage


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
    next: str
    messages: Annotated[list[AnyMessage], operator.add]
