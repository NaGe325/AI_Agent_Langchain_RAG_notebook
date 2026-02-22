from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os
load_dotenv()
os.environ["DASHSCOPE_API_KEY"] = os.getenv("APIKEY")


model = ChatTongyi(model="qwen3-max")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我给你提供已知参考资料为主，简洁和专业的方式来回答用户的问题。参考资料{context}。"),
        ("human", "用户的输入:{input}"),
    ]
)   

vector_store = InMemoryVectorStore(
    embedding=DashScopeEmbeddings(model = "text-embedding-v4")
)

#准备资料（向量库数据）
vector_store.add_texts(
    ["减肥就是少吃多练","跑步是很好的运动","吃水果有助于减肥"],
)

input_text = "减肥的好方法是什么？"

#根据输入的问题，从向量库中检索相关的资料
result= vector_store.similarity_search(
    input_text,
    k=2 
)
reference_text = "["
for doc in result:
    reference_text += doc.page_content
reference_text += "]"
#将检索到的资料拼接成一个字符串，作为上下文提供给模型

def print_prompt(prompt):
    print(prompt.to_string())
    return prompt

chain = prompt | print_prompt | model | StrOutputParser()

res = chain.invoke({"input":input_text, "context":reference_text})
print(res)