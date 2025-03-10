from typing_extensions import TypedDict


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
    documents: str
