import streamlit as st
from langserve import RemoteRunnable
from langchain_core.messages import HumanMessage
from PIL import Image


config = {"configurable": {"thread_id": "1"}}

ICON = Image.open("../assets/icon.ico")
st.set_page_config(
    page_title="Intelligent Multi-Agent ChatBot",
    layout="wide",
    page_icon=ICON,
    initial_sidebar_state="auto"
)

st.title("📊 Multi-Agent ChatBot")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


with st.spinner("🤔正在处理..."):
    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        try:
            app = RemoteRunnable("http://localhost:8000/Multi-Agent-ChatBot")
            message = [HumanMessage(content=prompt, name="user_chat")]
            input_all = {"messages": message,
                         "input": prompt,
                         "generation": "NULL",
                         "next": "NULL",
                         "documents": "NULL"
                         }

            responses = []
            for output in app.stream(input_all, config, stream_mode="values"):
                responses.append(output)

            for response in responses[::-1]:
                print(response)
                if response.get("chat", []):
                    last_response = response.get("chat", [])["generation"]
                    break
                elif response.get("generate", []):
                    last_response = response.get("generate", [])["generation"]
                    break
                elif response.get("arxiv_generate", []):
                    last_response = response.get("arxiv_generate", [])["generation"]
                    print(last_response)
                    print(type(last_response))
                    break
                else:
                    last_response = "Please ask again."

            with st.chat_message("assistant"):
                st.write(last_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": last_response})

            # 收缩显示 documents 的内容
            with st.expander("查看详细思考链信息"):
                st.write(responses)

        except Exception as e:
            st.error(f"处理时出现错误: {str(e)}")

# test demo: 你好，我叫XXX。   请问我叫什么名字？    你能帮我在bilibili上推荐几个有关 LangGraph 的视频吗？   请搜索几篇有关 KAN 的论文，并介绍每篇论文的摘要，最后给我论文的pdf地址。
# 请帮我在bilibili上推荐几个有关 LangGraph 的视频，并介绍视频的主要内容，并把视频地址和点赞数量等信息整理给我。
