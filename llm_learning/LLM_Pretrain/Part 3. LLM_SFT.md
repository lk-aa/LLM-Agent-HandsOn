# 微调数据构建的基本思路

## Q1：微调适用于哪些场景？我需要微调吗？

✅ 适合用微调的情况

| **场景** | **原因** |
|----------|--------|
| **让模型“记住”特定领域知识** | 比如法律、医疗、金融等特定行业，模型需要大量内部化的知识。 |
| **特定格式的输出** | 例如，代码补全、医学报告、法律合同等，输出必须严格符合行业格式。 |
| **领域专用术语或风格** | 例如，医生对话、律师咨询、金融分析师报告等，需遵循行业术语和专业表达。 |
| **任务特定的对话风格** | 例如 AI 助手需保持特定品牌语调，如 Apple Siri、微软 Copilot 的风格。 |

**⚠️ 何时不适合微调**
- **模型明显不需要学会、只是需要知道**：如某些规章制度、流程、公司内部机制、客观数据呈现等等。

> 医疗方面，诊断疾病、健康建议需要微调，医生排班、挂号流程、疾病分科、不需要微调。<br><br>
> 金融买卖策略、投资建议、行业分析需要微调，信息查询、实况数据不需要微调。<br><br>
> 金融发放贷款流程不需要微调、信贷评分不需要微调、出具信贷报告可以通过微调实现。<br><br>
> 数据分析流程中、生成SQL代码需要微调、获得数据后进行分析需要微调、获得数据的过程不需要微调。<br><br>
> **<font color="red">永远不要尝试让模型记住你的数据库，只能让模型学会并记住你的业务**</font>。
  
- **当知识频繁变化**：如新闻、法规等（微调后模型的知识不会自动更新）。
- **当训练数据不足**：如果数据量少，微调可能效果不佳，RAG 或 Prompt 更好。
- **成本过高时**：训练大模型需要 GPU 资源，不适用于小规模应用。

---

✅ 适合用 RAG 的情况
| **场景** | **原因** |
|----------|--------|
| **知识频繁更新** | 例如法律法规、公司政策、产品文档，RAG 可以随时查询最新信息。 |
| **数据量大但不适合微调** | 例如包含大量文档（百万级 FAQ、医学指南、学术论文），微调太昂贵但 RAG 可高效检索。 |
| **需要外部事实支撑** | 例如客户服务、技术支持，必须基于最新的文档或知识库回答。 |
| **个性化回答** | 例如查询用户历史数据、个性化推荐，RAG 可动态拉取用户相关信息。 |
| **特定领域的问答系统** | 例如企业内部 AI 助手，需要基于公司内部文档回答问题。 |

**⚠️ 何时不适合 RAG**
- **低延迟任务**：RAG 需要额外的检索步骤，比直接调用微调模型要慢。
- **复杂推理任务**：如果任务涉及深度推理（如多步推理、数学计算），仅靠检索可能不够。
- **小规模任务**：如果数据量很小，不如直接用 Prompt。

---

✅ 适合用 Prompt Engineering 的情况
| **场景** | **原因** |
|----------|--------|
| **短期测试 & 快速迭代** | 比如 MVP（最小可行产品）阶段，调整 Prompt 比微调更快。 |
| **非结构化任务** | 例如创意写作、头脑风暴、文案生成，Prompt 足以控制风格。 |
| **规则明确的任务** | 比如固定格式的输出（如“请按 JSON 格式输出”），无需微调。 |
| **无标注数据的任务** | 例如改进现有对话，不需要专门的训练数据。 |
| **多任务适应** | 例如 GPT-4 既能写代码，又能写论文，Prompt 可以让它适应不同任务。 |

**⚠️ 何时不适合 Prompt**
- **当任务需要严格控制风格或术语**（例如医疗报告，微调更好）。
- **当 Prompt 变得过长或复杂**（长 Prompt 会增加成本，也可能导致不稳定）。
- **当 API 调用成本过高**（如果需要大量 API 交互，微调可能更经济）。

---

## Q2：微调需要什么样的数据？

**微调数据是对话，也可以是文字/代码所构成的问答对（Q-A对）**。

**✅ 示例 1：医学问答微调（QA Fine-tuning）**
适用于医疗助手，让 AI 具备医学知识。

```jsonl
{"messages": [{"role": "system", "content": "你是一名专业的医学医生，擅长心血管疾病诊断。"},
              {"role": "user", "content": "什么是高血压？"},
              {"role": "assistant", "content": "高血压是一种长期疾病，指的是动脉血压长期高于正常范围，通常高于 140/90 mmHg。"}]}

{"messages": [{"role": "system", "content": "你是一名医学专家，专门解答关于糖尿病的问题。"},
              {"role": "user", "content": "糖尿病患者应该避免哪些食物？"},
              {"role": "assistant", "content": "糖尿病患者应避免高糖食品（如甜点、碳酸饮料）、高碳水化合物食品（如白米饭、面包）、高脂肪食品（如炸薯条、加工肉类）。"}]}
```

---

**✅ 示例 2：法律助理微调（Legal Assistant Fine-tuning）**
适用于法律 AI，帮助回答法律咨询。

```jsonl
{"messages": [{"role": "system", "content": "你是一名资深律师，专门解答合同法相关问题。"},
              {"role": "user", "content": "合同签署后是否可以反悔？"},
              {"role": "assistant", "content": "一般而言，合同一旦签署即具有法律效力，除非存在欺诈、胁迫、重大误解等特殊情况，才可以申请撤销。"}]}

{"messages": [{"role": "system", "content": "你是一名法律专家，擅长劳动法领域。"},
              {"role": "user", "content": "公司能随意解雇员工吗？"},
              {"role": "assistant", "content": "根据《劳动合同法》，公司不能无故解雇员工。必须提供合法理由，例如严重违反公司规章制度或合同到期不续签。"}]}
```

---

**✅ 示例 3：客户服务微调（Customer Support Fine-tuning）**
适用于 AI 客服，提升回答的准确性和品牌一致性。

```jsonl
{"messages": [{"role": "system", "content": "你是 XYZ 公司客户服务代表，回答要礼貌、专业。"},
              {"role": "user", "content": "请问如何申请退款？"},
              {"role": "assistant", "content": "您好，您可以在订单详情页点击‘申请退款’按钮，填写退款原因后提交。我们将在 3 个工作日内审核。"}]}

{"messages": [{"role": "system", "content": "你是 XYZ 品牌的客户服务代表，帮助用户解决订单问题。"},
              {"role": "user", "content": "我收到了错误的商品，怎么办？"},
              {"role": "assistant", "content": "很抱歉给您带来不便！请您提供订单号，我们将尽快安排换货或退款处理。"}]}
```

---

**✅ 示例 4：代码生成微调（Code Generation Fine-tuning）**
适用于代码助手，帮助自动生成代码或优化代码。

```jsonl
{"messages": [{"role": "system", "content": "你是一名 Python 开发专家，专门优化代码。"},
              {"role": "user", "content": "如何用 Python 读取 CSV 文件？"},
              {"role": "assistant", "content": "您可以使用 Pandas 读取 CSV 文件：\n```python\nimport pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())\n```"}]}

{"messages": [{"role": "system", "content": "你是 JavaScript 专家，擅长编写前端代码。"},
              {"role": "user", "content": "写一个 JavaScript 函数来计算数组中所有数字的和。"},
              {"role": "assistant", "content": "可以使用 `reduce()` 方法：\n```javascript\nfunction sumArray(arr) {\n    return arr.reduce((acc, num) => acc + num, 0);\n}\nconsole.log(sumArray([1, 2, 3, 4])); // 输出: 10\n```"}]}
```

---

**✅ 示例 5：摘要任务微调（Summarization Fine-tuning）**
适用于文档摘要生成，提高 AI 概括能力。

```jsonl
{"messages": [{"role": "system", "content": "你是一名专业的文章摘要助手，擅长提取关键内容。"},
              {"role": "user", "content": "以下是一个新闻报道，请用 50 字总结它。\n\n2024年，新型疫苗研发成功，全球疫情得到有效控制，多个国家宣布解除旅行限制……"},
              {"role": "assistant", "content": "2024年新疫苗成功研发，全球疫情受控，多个国家解除旅行限制，经济活动恢复。"}]}

{"messages": [{"role": "system", "content": "你是一个商业报告助手，可以帮助生成公司财报摘要。"},
              {"role": "user", "content": "以下是 2023 年 XYZ 公司的财报数据，请生成简要摘要。\n\nXYZ 公司 2023 年营收增长 15%，利润增长 8%，主要受新市场扩张推动……"},
              {"role": "assistant", "content": "XYZ 公司 2023 年营收增长 15%，利润增长 8%，主要因新市场扩张带动增长。"}]}
```

## Q3：怎样将各种各样的数据转换成JSONL格式？

**表格、sheet、数据不适合作为微调数据。如果你的数据是表格，则必须要配合相应的文字对话来训练**。

- **从数据库到JSONL**

如果你的数据存储在**数据库（DB）**中，并且不同的表之间有**关联（关系型数据库的外键、主键）**，那么你需要按照一定的逻辑**先整合数据（Raw Data），然后再转换为 JSONL 格式**。

**📌 1. 数据转换流程**
**步骤：**
1. **从数据库提取数据**
   - 连接数据库，查询相关表，按业务逻辑合并数据。
   - 处理多表关系（JOIN 操作）。
2. **转换为统一结构（Raw Data）**
   - 规范字段，使不同来源的数据保持一致性。
   - 处理缺失值、数据清理。
3. **映射到 JSONL 格式**
   - 组织成符合任务需求的 JSON 结构（QA、对话、摘要等）。
   - 确保每行数据是一个独立 JSON。

---

**✅ 场景 1：客户服务数据库 → JSONL（对话微调）**
**数据库结构（关系型数据库）**
- **customers**（客户信息表）：`id, name, email`
- **orders**（订单表）：`order_id, customer_id, product_name, status`
- **support_tickets**（客服工单）：`ticket_id, customer_id, issue, response`

**目标**：将客服历史对话转化为 JSONL 格式用于微调。

**🛠 SQL 查询（提取数据）**
```sql
SELECT c.name, s.issue, s.response
FROM customers c
JOIN support_tickets s ON c.id = s.customer_id;
```

**📌 转换为 JSONL**
```jsonl
{"messages": [{"role": "system", "content": "你是一名客户服务 AI，帮助用户解决订单问题。"},
              {"role": "user", "content": "我收到了错误的商品，怎么办？"},
              {"role": "assistant", "content": "很抱歉给您带来不便！请提供订单号，我们将安排换货或退款处理。"}]}
```

**🛠 Python 代码**
```python
import json
import sqlite3  # 适用于 SQLite，其他数据库可以使用 MySQL 或 PostgreSQL 连接

# 连接数据库
conn = sqlite3.connect("database.db")
cursor = conn.cursor()

# 查询数据
query = """
SELECT c.name, s.issue, s.response
FROM customers c
JOIN support_tickets s ON c.id = s.customer_id
"""
cursor.execute(query)
data = cursor.fetchall()

# 转换 JSONL
jsonl_data = []
for row in data:
    user_message = row[1]  # 客户问题
    assistant_response = row[2]  # AI 期望的回复
    json_obj = {
        "messages": [
            {"role": "system", "content": "你是一名客户服务 AI，帮助用户解决订单问题。"},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ]
    }
    jsonl_data.append(json_obj)

# 保存 JSONL 文件
with open("customer_support.jsonl", "w", encoding="utf-8") as f:
    for obj in jsonl_data:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("✅ JSONL 数据已成功转换！")
```

---

**✅ 场景 2：医学数据库 → JSONL（QA 训练数据）**
**数据库结构**
- **patients**（病人信息）：`id, name, age, condition`
- **medical_cases**（病历记录）：`case_id, patient_id, diagnosis, treatment`
- **faq**（医学问答知识库）：`question, answer`

**目标**：将病历 + FAQ 结合，转换为 JSONL 格式，供 AI 训练医学问答能力。

**🛠 SQL 查询**
```sql
SELECT f.question, f.answer
FROM faq f
UNION
SELECT m.diagnosis, m.treatment
FROM medical_cases m;
```

**📌 转换为 JSONL**
```jsonl
{"messages": [{"role": "system", "content": "你是一名专业医生，擅长诊断和治疗常见疾病。"},
              {"role": "user", "content": "感冒应该如何治疗？"},
              {"role": "assistant", "content": "感冒通常是病毒感染引起的，可以通过多喝水、休息、适当服用退烧药来缓解症状。"}]}

{"messages": [{"role": "system", "content": "你是一名医生，帮助分析病人病例并提供治疗建议。"},
              {"role": "user", "content": "病人 45 岁，确诊高血压，应该如何治疗？"},
              {"role": "assistant", "content": "建议低盐饮食、适量运动，并在医生指导下服用降压药，如氨氯地平或依那普利。"}]}
```

**🛠 Python 代码**
```python
import json
import sqlite3

# 连接数据库
conn = sqlite3.connect("medical.db")
cursor = conn.cursor()

# 查询 FAQ 和病例数据
query = """
SELECT 'FAQ' AS source, question, answer FROM faq
UNION
SELECT 'Case' AS source, diagnosis, treatment FROM medical_cases;
"""
cursor.execute(query)
data = cursor.fetchall()

# 转换 JSONL
jsonl_data = []
for row in data:
    user_message = row[1]  # 问题/病例描述
    assistant_response = row[2]  # 回答/治疗方案
    json_obj = {
        "messages": [
            {"role": "system", "content": "你是一名医学专家，专门解答健康和治疗问题。"},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ]
    }
    jsonl_data.append(json_obj)

# 保存 JSONL 文件
with open("medical_data.jsonl", "w", encoding="utf-8") as f:
    for obj in jsonl_data:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("✅ JSONL 数据已成功转换！")
```

---

**📌 3. 数据库 → JSONL 的转换要点**
1. **理解数据关系**
   - 如果数据库有多个表，使用 `JOIN` 语句整合数据。
   - 根据业务需求，提取 `问题` 和 `回答`，确保数据清晰。

2. **数据清理**
   - 处理 `NULL` 值（用默认值填充）。
   - 统一数据格式，如 `日期格式` 转换。

3. **批量转换**
   - **小数据**：可以直接查询后 `json.dumps()` 转 JSONL。
   - **大数据（百万级）**：
     - 分批查询（分页 `LIMIT OFFSET`）。
     - 用 **流式写入 JSONL** 避免内存占用过高。

## Q4：没有对话信息怎么办？

如果你没有现成的对话库，只有数据库中的**原始数据**（如医疗病例、产品信息、法律条款等），你可以**合成微调数据集**，主要方法有两种：

1. **规则生成（Rule-based Generation）**：适用于格式较固定、可程序化生成的场景。
2. **大语言模型（LLM 合成）**：适用于需要自然语言表达、需要更多样化数据的场景（如 GPT-4 生成问答对）。

---

**📌 方法 1：规则生成（Rule-based Generation）**
- 你的数据库数据结构清晰，比如**病人信息、产品FAQ、合同条款**等。
- 你能定义固定的**问题模板**，程序化生成问答。

**📌 示例 1：医疗数据库 → 规则生成对话数据**

**数据库（Medical DB）示例**
| patient_id | diagnosis     | treatment                |
|------------|--------------|--------------------------|
| 101        | 高血压       | 低盐饮食、服用降压药     |
| 102        | 糖尿病       | 控制血糖、注射胰岛素     |
| 103        | 哮喘         | 使用吸入剂、避免过敏原   |

**合成问答规则**
- **规则**：
  - 问题：`"患者被诊断为 {diagnosis}，应如何治疗？"`
  - 回答：`"{treatment}"`

**Python 代码**
```python
import json

# 假设从数据库中提取的原始数据
medical_cases = [
    {"diagnosis": "高血压", "treatment": "低盐饮食、服用降压药"},
    {"diagnosis": "糖尿病", "treatment": "控制血糖、注射胰岛素"},
    {"diagnosis": "哮喘", "treatment": "使用吸入剂、避免过敏原"},
]

jsonl_data = []

for case in medical_cases:
    question = f"患者被诊断为 {case['diagnosis']}，应如何治疗？"
    answer = case["treatment"]

    json_obj = {
        "messages": [
            {"role": "system", "content": "你是一名医生，专门提供治疗建议。"},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    }
    jsonl_data.append(json_obj)

# 保存 JSONL
with open("synthetic_medical_data.jsonl", "w", encoding="utf-8") as f:
    for obj in jsonl_data:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("✅ 规则生成的 JSONL 数据已完成！")
```

**生成 JSONL 数据**
```jsonl
{"messages": [{"role": "system", "content": "你是一名医生，专门提供治疗建议。"},
              {"role": "user", "content": "患者被诊断为 高血压，应如何治疗？"},
              {"role": "assistant", "content": "低盐饮食、服用降压药"}]}

{"messages": [{"role": "system", "content": "你是一名医生，专门提供治疗建议。"},
              {"role": "user", "content": "患者被诊断为 糖尿病，应如何治疗？"},
              {"role": "assistant", "content": "控制血糖、注射胰岛素"}]}
```

✅ **优点**：
- 规则清晰，数据可靠，适用于**结构化数据**。
- 不依赖大模型，成本低，适用于**大批量数据生成**。

❌ **缺点**：
- 生成的问答可能**缺乏多样性**，显得死板。
- 适用于结构化数据，不适合复杂问答（如多轮对话）。

---

**📌 方法 2：大语言模型（LLM 合成）**
- 你希望生成**更自然、更丰富的对话**。
- 数据库没有标准问答，但有很多**有价值的信息**（如合同条款、产品手册）。
- 你可以调用 GPT-4 或其他 LLM 来帮助生成对话数据。

---

**📌 示例 2：法律数据库 → LLM 生成对话**
**数据库（Legal DB）示例**
| law_id | law_title            | law_content                                 |
|--------|----------------------|---------------------------------------------|
| 201    | 劳动合同法 第 36 条  | 劳动者每日工作时间不得超过 8 小时。         |
| 202    | 消费者权益保护法 第 24 条 | 经营者提供的商品或服务如有瑕疵，消费者可要求退货。 |

**使用 GPT-4 生成问答**

- **Prompt（提示词）**：

```
你是一名法律专家。请根据以下法律条款生成 3 组高质量的问答对：
---
劳动合同法 第 36 条：
劳动者每日工作时间不得超过 8 小时。
---
请生成问答：
```

**Python 代码**
```python
import json
import openai  # 需要 OpenAI API

api_key = "your-openai-api-key"

legal_cases = [
    {"law_title": "劳动合同法 第 36 条", "law_content": "劳动者每日工作时间不得超过 8 小时。"},
    {"law_title": "消费者权益保护法 第 24 条", "law_content": "经营者提供的商品或服务如有瑕疵，消费者可要求退货。"},
]

jsonl_data = []

for case in legal_cases:
    prompt = f"""
    你是一名法律专家。请根据以下法律条款生成 3 组高质量的问答对：
    ---
    {case['law_title']}：
    {case['law_content']}
    ---
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "你是法律专家，擅长生成法律问答数据。"},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500,
        api_key=api_key
    )

    qa_pairs = response["choices"][0]["message"]["content"].strip().split("\n\n")

    for qa in qa_pairs:
        if "Q:" in qa and "A:" in qa:
            question = qa.split("Q: ")[1].split("\nA: ")[0]
            answer = qa.split("\nA: ")[1]
            json_obj = {
                "messages": [
                    {"role": "system", "content": "你是一名法律专家，帮助回答法律相关问题。"},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]
            }
            jsonl_data.append(json_obj)

# 保存 JSONL
with open("synthetic_legal_data.jsonl", "w", encoding="utf-8") as f:
    for obj in jsonl_data:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("✅ LLM 生成的 JSONL 数据已完成！")
```

**生成 JSONL 数据**
```jsonl
{"messages": [{"role": "system", "content": "你是一名法律专家，帮助回答法律相关问题。"},
              {"role": "user", "content": "根据劳动合同法，员工每天最多能工作多少小时？"},
              {"role": "assistant", "content": "根据《劳动合同法》第 36 条，劳动者每日工作时间不得超过 8 小时。"}]}
```

✅ **优点**：
- 问答多样化，语义更自然，适用于**开放性任务**。
- 可以模拟**真实用户的提问方式**，提高训练效果。

❌ **缺点**：
- 依赖 GPT-4 API，有**调用成本**。
- 可能生成**错误或偏差信息**，需要人工检查。

---
