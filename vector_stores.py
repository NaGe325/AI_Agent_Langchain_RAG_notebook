from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import DashScopeEmbeddings
#这里根据不同的数据类型选择不同的loader，文本文件使用TextLoader，PDF文件使用PyPDFLoader，CSV文件使用CSVLoader
from langchain_community.document_loaders import CSVLoader
vector_store = InMemoryVectorStore(
    embedding_function=DashScopeEmbeddings()
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
    k=5 #根据查询语句进行相似度搜索，返回与查询语句最相似的前k个文档，这里返回前5个文档
) 

print(result) #打印搜索结果