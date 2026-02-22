from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document 

from dotenv import load_dotenv
import os
from langchain_core.runnables import RunnablePassthrough
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

#langchain中向量存储对象，有一个方法叫做as_retriever()，可以返回一个Runnable接口的子类实例对象，将向量存储对象转换为一个检索器对象，这个检索器对象有一个方法叫做get_relevant_documents()，可以根据输入的查询语句来检索相关的文档，并返回一个文档列表
retriever = vector_store.as_retriever(search_kwargs={"k":2}) #这里指定了检索的参数k，表示返回与查询语句最相似的前k个文档，这里返回前2个文档

def format_func(docs:list[Document]):   
    if not docs:
        return "没有检索到相关资料"
    formatted_context = "["
    for doc in docs:
        formatted_context += doc.page_content
    formatted_context += "]"
    return formatted_context

chain = (
    {"input": RunnablePassthrough(), "context": retriever | format_func} | prompt | model | StrOutputParser()
)

res = chain.invoke(input_text)
'''
retriever : 
输入是用户提问 str
输出是向量库的检索结果 list[Document]

prompt :
输入是用户提问+向量库检索结果 dict
输出是完整的提示词 PromptValue

'''
print(res)

