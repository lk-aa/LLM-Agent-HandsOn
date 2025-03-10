from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage


coder_llm = ChatOllama(
    base_url="http://localhost:11434",          # 注意:这里需要替换成自己本地启动的endpoint地址
    model="llama3.2:3b",
)

print(coder_llm.invoke("请你使用Python实现三着色问题。").content)
