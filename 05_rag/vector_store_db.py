from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
#这里根据不同的数据类型选择不同的loader，文本文件使用TextLoader，PDF文件使用PyPDFLoader，CSV文件使用CSVLoader
from langchain_community.document_loaders import CSVLoader

from dotenv import load_dotenv
import os
load_dotenv()
os.environ["DASHSCOPE_API_KEY"] = os.getenv("APIKEY")


#Chroma 向量数据库（轻量级别的）
#确保有安装langchain-chroma, chromadb库：pip install chromadb
vector_store = Chroma(
    collection_name="qa_collection", #指定集合名称
    embedding_function=DashScopeEmbeddings(), #指定使用的嵌入函数，这里使用DashScopeEmbeddings
    persist_directory="./chroma_db" #指定持久化存储的目录，这里存储在当前目录下的chroma_db文件夹中
)

loader = CSVLoader(
    file_path="./data/qa.csv",
    encoding="utf-8",
    source_column="source" #指定数据来源的列名
)

document = loader.load() #加载数据

#vector add document
vector_store.add_documents(
    documents=document, # 将加载的数据添加到向量存储中,类型为list[Document]
    ids=["id"+str(i) for i in range(1,len(document)+1)] #为每个文档生成一个唯一的id，这里使用简单的数字id
)

vector_store.delete(ids=["id1"]) #根据id删除文档，这里删除id为id1的文档

result = vector_store.similarity_search(
    query="什么是人工智能？",
    k=5, #根据查询语句进行相似度搜索，返回与查询语句最相似的前k个文档，这里返回前5个文档
    filter={"source": "qa.csv"} #根据数据来源进行过滤，这里只搜索数据来源为qa.csv的文档
)

print(result) #打印搜索结果