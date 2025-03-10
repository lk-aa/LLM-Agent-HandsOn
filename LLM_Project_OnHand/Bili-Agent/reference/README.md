# 实践与参考

## 1. BiliAgent

这是一个关于推荐b站视频的Agent。用户输入需求，由大模型进行理解，提取出需要查询的query，再利用bili-api工具，根据query进行视频查询，最后返回查询结果。根据结果构建RAG，并提取出关键信息。再由Agent分别判断每个视频与用户问题的相关性等问题，最后回答用户问题。\
框架：LangGraph, LangChain, FastAPI

## 2. ReAct_AI_Agent

这是利用提示工程和ReAct机制构建的客服Agent, 其中仅涉及基础大模型的调用。工具调用, 思考过程传递等是由开发者手动实现的, 未使用Agent框架。

## 3. game_builder_crew

一个Crewai框架构建的Multi-Agent, 能够实现游戏的策划, 代码编写, 代码优化等过程。（目前效果一般）

## 4. surprise_travel

一个Crewai框架构建的Multi-Agent, 能够完成互联网查询等功能，进行旅游策划。
