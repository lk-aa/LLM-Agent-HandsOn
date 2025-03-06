# 项目中 DeepSeek 和 Ollama 服务接口详细报告

## 一、项目概述
DS-Agent 是一个基于大语言模型构建的智能客服系统，支持多种大语言模型，其中包括 DeepSeek 和 Ollama。系统提供了丰富的功能，如聊天、推理、搜索等，用户可以与系统进行实时交互。本报告将详细介绍 DeepSeek 和 Ollama 服务的接口，包括接口的功能、输入输出参数以及代码实现。

## 二、DeepSeek 服务接口

### 2.1 初始化
```python
class DeepseekService:
    def __init__(self, model: str = "deepseek-chat"):
        logger.info("Initializing Deepseek Service")
        self.client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL
        )
        self.model = settings.DEEPSEEK_MODEL or model 
        self.cache = RedisSemanticCache(prefix="deepseek")
```
- **功能**：初始化 DeepSeek 服务，设置 API 密钥、基础 URL、模型名称，并初始化 Redis 语义缓存。
- **参数**：
  - `model`：模型名称，默认为 "deepseek-chat"。

### 2.2 流式生成回复
```python
async def generate_stream(
    self, 
    messages: List[Dict],
    user_id: Optional[int] = None,
    conversation_id: Optional[int] = None,
    on_complete: Optional[Callable[[int, int, List[Dict], str], None]] = None
) -> AsyncGenerator[str, None]:
    try:
        cache = RedisSemanticCache(prefix="deepseek", user_id=user_id)
        start_time = time.time()
        cached_response = await cache.lookup(messages)
        if cached_response:
            response_time = time.time() - start_time
            logger.info(f"Cache hit! Response time: {response_time:.4f} seconds")
            async for chunk in self._stream_cached_response(cached_response):
                yield chunk
            if on_complete and user_id is not None and conversation_id is not None:
                await on_complete(user_id, conversation_id, messages, cached_response)
            return
        full_response = []
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                full_response.append(chunk.choices[0].delta.content)
                content = json.dumps(chunk.choices[0].delta.content, ensure_ascii=False)
                yield f"data: {content}\n\n"
        complete_response = "".join(full_response)
        await cache.update(messages, complete_response)
        response_time = time.time() - start_time
        logger.info(f"Cache miss. Response time: {response_time:.4f} seconds")
        if on_complete and user_id is not None and conversation_id is not None:
            await on_complete(user_id, conversation_id, messages, complete_response)
    except Exception as e:
        logger.error(f"Error in generate_stream: {str(e)}", exc_info=True)
        error_msg = json.dumps(f"生成回复时出错: {str(e)}", ensure_ascii=False)
        yield f"data: {error_msg}\n\n"
```
- **功能**：流式生成回复，先检查缓存，如果缓存命中则返回缓存结果，否则调用 DeepSeek API 生成回复，并将结果存入缓存。
- **输入参数**：
  - `messages`：对话消息列表，每个消息为一个字典。
  - `user_id`：用户 ID，可选参数。
  - `conversation_id`：会话 ID，可选参数。
  - `on_complete`：完成回调函数，可选参数。
- **输出**：异步生成器，返回流式回复。

### 2.3 非流式生成回复
```python
async def generate(self, messages: List[Dict]) -> str:
    try:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Generation error: {str(e)}")
        raise
```
- **功能**：非流式生成回复，直接调用 DeepSeek API 生成完整回复。
- **输入参数**：
  - `messages`：对话消息列表，每个消息为一个字典。
- **输出**：完整的回复内容。

## 三、Ollama 服务接口

### 3.1 初始化
```python
class OllamaService:
    def __init__(self):
        logger.info("Initializing Ollama Service")
        self.base_url = settings.OLLAMA_BASE_URL
        self.chat_model = settings.OLLAMA_CHAT_MODEL
        self.reason_model = settings.OLLAMA_REASON_MODEL
```
- **功能**：初始化 Ollama 服务，设置基础 URL、聊天模型和推理模型。
- **参数**：无

### 3.2 流式生成回复
```python
async def generate_stream(
    self, 
    messages: List[Dict],
    user_id: Optional[int] = None,
    conversation_id: Optional[int] = None,
    on_complete: Optional[Callable] = None
) -> AsyncGenerator[str, None]:
    try:
        model = self.reason_model
        logger.info(f"Using model: {model}")
        full_response = []
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "keep_alive": -1,
                    "options": {
                        "temperature": 0.7,
                    }
                }
            ) as response:
                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line)
                            if content := chunk.get("message", {}).get("content"):
                                full_response.append(content)
                                content = json.dumps(content, ensure_ascii=False)
                                yield f"data: {content}\n\n"
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {str(e)}")
                            continue
        if on_complete:
            complete_response = "".join(full_response)
            await on_complete(user_id, conversation_id, messages, complete_response)
    except Exception as e:
        logger.error(f"Stream generation error: {str(e)}")
        error_msg = json.dumps(f"生成回复时出错: {str(e)}", ensure_ascii=False)
        yield f"data: {error_msg}\n\n"
```
- **功能**：流式生成回复，使用推理模型调用 Ollama 的 `/api/chat` 接口，处理流式响应。
- **输入参数**：
  - `messages`：对话消息列表，每个消息为一个字典。
  - `user_id`：用户 ID，可选参数。
  - `conversation_id`：会话 ID，可选参数。
  - `on_complete`：完成回调函数，可选参数。
- **输出**：异步生成器，返回流式回复。

### 3.3 非流式生成回复
```python
async def generate(self, messages: List[Dict]) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.chat_model,
                    "messages": messages,
                    "stream": False,
                    "keep_alive": -1,
                    "options": {
                        "temperature": 0.7,
                    }
                }
            ) as response:
                result = await response.json()
                return result["message"]["content"]
    except Exception as e:
        print(f"Generation error: {str(e)}")
        raise
```
- **功能**：非流式生成回复，使用聊天模型调用 Ollama 的 `/api/chat` 接口，返回完整回复。
- **输入参数**：
  - `messages`：对话消息列表，每个消息为一个字典。
- **输出**：完整的回复内容。

## 四、LLM 工厂选择服务
```python
class LLMFactory:
    @staticmethod
    def create_chat_service():
        if settings.CHAT_SERVICE == ServiceType.DEEPSEEK:
            return DeepseekService()
        else:
            return OllamaService()
```
- **功能**：根据配置文件中的 `CHAT_SERVICE` 设置，选择使用 DeepSeek 服务或 Ollama 服务。
- **参数**：无
- **输出**：`DeepseekService` 或 `OllamaService` 实例。

## 五、总结
### 5.1 接口特点
- **DeepSeek 服务**：支持缓存机制，提高响应速度，使用 OpenAI 风格的 API 调用方式。
- **Ollama 服务**：支持流式和非流式回复，使用自定义的 `/api/chat` 接口，可配置不同的模型用于聊天和推理。
- **LLM 工厂**：提供了灵活的服务选择机制，方便根据配置切换不同的大语言模型服务。

### 5.2 改进建议
- **错误处理**：可以进一步完善错误处理机制，提供更详细的错误信息，方便调试和排查问题。
- **性能优化**：可以对缓存机制进行优化，提高缓存命中率，减少 API 调用次数。
- **功能扩展**：可以增加更多的接口功能，如模型选择、参数调整等，以满足不同用户的需求。