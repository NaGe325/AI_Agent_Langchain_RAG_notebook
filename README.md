# AI Agent & RAG 学习实战 (基于黑马程序员教程)

本项目是跟随 B站 **黑马程序员** 的《大模型RAG与Agent智能体项目实战》教程的学习实践代码仓库。主要基于 **LangChain** 框架和 **阿里云百炼 (通义千问)** 大模型，从基础的提示词工程到 RAG (检索增强生成) 的完整实现。

## 🛠️ 技术栈

- **语言**: Python 3.10+
- **框架**: LangChain (Core, Community)
- **大模型**: 阿里云通义千问 (DashScope / Qwen)
- **向量数据库**: ChromaDB
- **Embedding**: DashScope Text Embedding

## 🚀 快速开始

### 1. 环境准备

建议使用 `conda` 或 `venv` 创建虚拟环境。

```bash
# 创建虚拟环境
python -m venv .venv

# 激活环境 (Mac/Linux)
source .venv/bin/activate

# 激活环境 (Windows)
.venv\Scripts\activate
```

### 2. 安装依赖

请确保安装了以下核心依赖库：

```bash
pip install langchain langchain-community langchain-core dashscope chromadb python-dotenv
```

### 3. 配置环境变量

在项目根目录下创建一个 `.env` 文件，并填入你的阿里云 DashScope API Key：

```properties
# .env 文件内容
APIKEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
> 注意：代码中会自动读取 `APIKEY` 并将其设置为 LangChain 和 DashScope 所需的 `DASHSCOPE_API_KEY` 环境变量。

## 📂 项目目录结构说明

本项目代码按照功能模块进行了分类整理：

### 📁 00_basics (基础入门)
- `test_api_key.py`: 测试阿里云 API Key 配置。
- `test_openAI.py`: 使用 OpenAI 兼容接口调用通义千问。
- `test_json_basics.py`: JSON 解析基础测试。

### 📁 01_models (模型调用)
- `langchain_aliyun_llms.py`: 基础 LLM 调用示例。
- `langchain_aliyun_chat_model.py`: Chat Model (对话模型) 调用示例。
- `langchain_aliyun_embedding_model.py`: 文本向量化 (Embedding) 模型测试。

### 📁 02_prompts (提示词工程)
- `langchain_prompt_template.py`: 基础提示词模板 (`PromptTemplate`) 使用。
- `langchain_prompt_chat.py`: 聊天提示词模板 (`ChatPromptTemplate`) 使用。
- `langchain_prompt_fewshot.py`: 少样本提示词 (`FewShotPromptTemplate`) 示例。
- `test_prompt.py`: 提示词测试脚本。

### 📁 03_parsers (输出解析)
- `langchain_StrOutputParser.py`: 字符串输出解析器。
- `langchain_JsonOutputParser.py`: JSON 格式输出解析器。

### 📁 04_memory (记忆机制)
- `memory_temporary.py`: 临时对话记忆 (`InMemoryChatMessageHistory`)。
- `memory_long.py`: 长期记忆 (`FileChatMessageHistory`)。
- `langchain_chains.py`: 带有历史记录的对话链示例。

### 📁 05_rag (检索增强生成)
- `vector_store_db.py`: 初始化向量数据库 (ChromaDB) 并存储数据。
- `vector_stores.py`: 向量数据库的基础操作测试。
- `vector_store_RunnablePassthrough.py`: 使用 LCEL 构建完整的 RAG 问答链。
- `vector_store_prompt_online.py`: 结合在线 Prompt 的 RAG 实现。

### 📁 docs & data
- `docs/`: 存放项目相关的文档 (如 PDF 教程)。
- `data/`: 存放运行时产生的数据 (如 `chroma_db` 向量库, `chat_history` 聊天记录)。

## 📝 学习笔记

- **API Key 管理**: 统一使用 `.env` 文件管理，通过 `python-dotenv` 加载，避免 Key 泄露。
- **LangChain LCEL**: 项目中大量使用了 LangChain 的声明式表达语言 (Runnables)，如 `chain = prompt | model | parser`，代码更加简洁易读。
- **RAG 流程**: 
  1. **Load**: 加载各类文档 (PDF/Txt/CSV)。
  2. **Split**: 文本分割。
  3. **Embed**: 使用 Embedding 模型向量化。
  4. **Store**: 存入 ChromaDB。
  5. **Retrieve**: 根据问题检索相关片段。
  6. **Generate**: LLM 根据检索到的上下文回答问题。

## 🔗 参考资料

- [LangChain 官方文档](https://python.langchain.com/docs/get_started/introduction)
- [阿里云 DashScope 文档](https://help.aliyun.com/zh/dashscope/developer-reference/api-details)
- [黑马程序员 B站教程](https://www.bilibili.com/)


