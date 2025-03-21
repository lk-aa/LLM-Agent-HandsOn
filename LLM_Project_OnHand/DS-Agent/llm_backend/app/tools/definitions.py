"""工具描述定义文件"""

SEARCH_TOOL = {
    "name": "search",
    "description": "使用谷歌搜索从互联网获取更多的实时信息",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "通过搜索从互联网获取的信息的问题、内容、关键词等"
            }
        },
        "required": ["query"]
    }
}

ARXIVSEARCH_TOOL = {
    "name": "arxiv_search",
    "description": "使用Arxiv网站的API查询指定主题的论文",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "输入给Arxiv网站的API获取查询论文信息的问题、内容、关键词等"
            }
        },
        "required": ["query"]
    }
}

BILISEARCH_TOOL = {
    "name": "bili_search",
    "description": "使用Bilibili网站的API查询指定内容的信息",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "输入给Bilibili网站的API获取查询结果的问题、内容、关键词等"
            }
        },
        "required": ["query"]
    }
}

# # 可以添加更多工具定义
# WEATHER_TOOL = {
#     "name": "get_weather",
#     "description": "获取天气信息",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "city": {
#                 "type": "string",
#                 "description": "城市名称"
#             }
#         },
#         "required": ["city"]
#     }
# }

# 工具定义集合
TOOL_DEFINITIONS = {
    "search": SEARCH_TOOL,
    "arxiv_search": ARXIVSEARCH_TOOL,
    "bili_search": BILISEARCH_TOOL,
    # "weather": WEATHER_TOOL
} 