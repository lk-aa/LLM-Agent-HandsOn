from typing import TypedDict, Literal


members = ["chat", "bili_analysis", "arxiv_retriever"]
options = members + ["FINISH"]


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH"""
    next: Literal[*options]
