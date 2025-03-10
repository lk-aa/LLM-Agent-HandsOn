import ollama

# 生成回答并逐字符打印
stream = ollama.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': '天为什么是绿的'}],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)


def test_ollama_chat():
    # Test case 1: Test with a simple question
    messages1 = [{'role': 'user', 'content': '天为什么是绿的'}]
    stream1 = ollama.chat(model='llama3.2', messages=messages1, stream=True)
    response_content = ""
    for chunk in stream1:
        response_content += chunk['message']['content']
    # 检查回答是否合理
    assert "绿色" in response_content or "为什么" in response_content
    print("All test cases pass")


test_ollama_chat()
