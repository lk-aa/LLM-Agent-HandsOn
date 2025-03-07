# 项目技术文档和报告

## 一、项目整体叙述
### 1.1 项目概述
- **项目名称**：DS-Agent - 基于大语言模型构建的智能客服系统
- **项目简介**：DS-Agent 是一个基于大语言模型构建的智能客服系统，旨在为用户提供智能问答、推理、搜索以及文件处理等功能。系统支持用户注册和登录，用户可以与系统进行实时交互，系统能够根据用户的输入提供相应的回答和解决方案。同时，系统还具备文件上传和基于文档的问答功能，方便用户处理和查询相关文档信息。具体地，其具备多种功能特性，如深度思考能力、流式问答、实时联网检索和本地知识库问答等。

### 1.2 技术栈概述
- **后端**：采用 FastAPI 作为 Web 框架，结合 SQLAlchemy 进行数据库操作，使用 MySQL 作为数据库存储数据。支持 Ollama 和 DeepSeek 等大语言模型，提供聊天、推理、搜索等服务。
    - FastAPI：是一个高性能的 Python Web 框架，用于构建后端 API。它基于 > Python 的类型提示，提供了快速开发和高效运行的能力。
    - SQLAlchemy：是一个强大的 Python SQL 工具包，提供了统一的数据库操作接口，支持多种数据库，如 MySQL。在项目中，可能用于与 MySQL 数据库进行交互，实现数据的增删改查操作。
    - MySQL：是一种开源的关系型数据库管理系统，用于存储项目中的数据，如用户信息、问答记录等。
    - Ollama/DeepSeek：Ollama 可能是用于管理和部署模型的工具，而 DeepSeek 是大语言模型，项目支持使用 Ollama 接入 Deepseek r1 模型系列，并且可以使用 DeepSeek R1 的在线 API 实现深度思考功能。

- **前端**：使用 Vue 3、Element Plus 和 TypeScript 构建用户界面，提供友好的用户交互体验。
    - Vue 3：是一个流行的 JavaScript 前端框架，用于构建用户界面。它具有响应式数据绑定、组件化开发等特性，能够提高开发效率和代码可维护性。
    - Element Plus：是基于 Vue 3 的 UI 组件库，提供了丰富的组件，如按钮、表单、表格等，方便快速搭建美观的用户界面。
    - TypeScript：是 JavaScript 的超集，为 JavaScript 增加了静态类型检查，能够提高代码的可靠性和可维护性。

## 二、功能特性介绍
### 2.1 用户注册和登录
1. 用户可以通过注册和登录功能来使用 DS-Agent 系统。注册时，用户需要提供用户名、邮箱、密码和确认密码等信息，系统会对输入进行验证并将用户信息存储在数据库中。登录时，用户需要提供邮箱和密码，系统会验证用户信息并生成访问令牌，供后续的请求使用。
2. 用户信息存储在数据库中，包括用户名、邮箱、密码等信息。

### 2.2 流式问答功能
1. **DeepSeek v3 & Ollama + 问答类模型（如 qwen2.5）**：系统支持使用 DeepSeek v3 和 Ollama 接入问答类模型，实现流式问答功能，提高用户体验。
2. **DeepSeek R1 & Ollama + Deepseek r1**：同样支持使用 DeepSeek R1 和 Ollama 接入 Deepseek r1 模型，实现深度思考流式问答。

### 2.3 深度思考能力
1. 支持 DeepSeek R1 在线 API：通过调用 DeepSeek R1 的在线 API，系统可以实现深度思考功能，为用户提供更准确和深入的回答。
2. 支持使用 Ollama 接入任意 Deepseek r1 模型系列：用户可以使用 Ollama 工具接入不同的 Deepseek r1 模型，实现灵活的模型配置。
3. 灵活的模型配置：用户可以根据需求选择不同的模型，以满足不同场景的应用需求。

### 2.4 实时联网检索
1. Deepseek v3 + Serper API：通过结合 Deepseek v3 模型和 Serper API，系统可以实现实时联网检索功能，为用户提供最新的信息。

### 2.5 本地知识库问答
1. Deepseek v3 + sentence-transformers：利用 Deepseek v3 模型和 sentence-transformers 技术，系统可以实现本地知识库问答功能，根据本地知识库中的信息回答用户的问题。


## 三、后端技术详细介绍

### 3.1 FastAPI 应用初始化 && 路由接口
详细内容参考文件`api.md`.

### 3.3 数据库表结构设计
项目中使用 SQLAlchemy 进行数据库操作，数据库表结构包括：
- **用户表**：存储用户的基本信息，如用户名、密码、邮箱等。
- **会话表**：存储用户的会话信息，如会话 ID、用户 ID、会话名称等。
- **消息表**：存储用户与系统的聊天消息，如消息 ID、会话 ID、用户 ID、消息内容等。

#### 3.3.1 用户表 (Users)

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | Integer | PRIMARY KEY | 用户唯一标识 |
| username | String(50) | UNIQUE, NOT NULL | 用户名 |
| email | String(100) | UNIQUE, NOT NULL | 电子邮件 |
| password_hash | String(255) | NOT NULL | 密码哈希值 |
| created_at | DateTime | DEFAULT NOW() | 创建时间 |
| updated_at | DateTime | DEFAULT NOW() | 更新时间 |
| last_login | DateTime | NULL | 最后登录时间 |
| status | String(20) | DEFAULT 'active' | 用户状态 |

#### 3.3.2 会话表 (Conversations)

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | Integer | PRIMARY KEY | 对话唯一标识 |
| user_id | Integer | FOREIGN KEY | 关联用户ID |
| created_at | DateTime | DEFAULT NOW() | 创建时间 |
| updated_at | DateTime | DEFAULT NOW() | 更新时间 |
| status | String(20) | DEFAULT 'ongoing' | 对话状态 |
| dialogue_type | Enum | NOT NULL | 对话类型 |


#### 3.3.3 消息表 (Messages)

| 字段名 | 类型 | 约束 | 说明 |
|--------|------|------|------|
| id | Integer | PRIMARY KEY | 消息唯一标识 |
| conversation_id | Integer | FOREIGN KEY | 关联对话ID |
| sender | String(50) | NOT NULL | 发送者 |
| content | Text | NOT NULL | 消息内容 |
| created_at | DateTime | DEFAULT NOW() | 创建时间 |
| message_type | String(20) | DEFAULT 'text' | 消息类型 |



### 3.4 大语言模型服务
在 `./llm_backend/app/services/...` 文件代码中，实现了 LLM 的相关服务, 详细内容参考文件`llm_api.md`.

## 四、项目总结
### 5.1 项目亮点
- **高性能**：采用 FastAPI 框架，结合异步编程和流式响应，提高系统的处理能力和响应速度。
- **多语言模型支持**：支持 Ollama 和 DeepSeek 等大语言模型，为用户提供多样化的服务。
- **文件处理和 RAG 功能**：支持文件上传和基于文档的问答功能，方便用户处理和查询相关文档信息。
- **用户认证和会话管理**：提供用户注册、登录和会话管理功能，保障用户信息安全。
