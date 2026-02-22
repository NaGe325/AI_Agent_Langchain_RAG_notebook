from langchain_community.embeddings import DashScopeEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["DASHSCOPE_API_KEY"] = os.getenv("APIKEY")

model = DashScopeEmbeddings()

#embed_query,embed_documents
print(model.embed_query("hello world"))
print(model.embed_documents(["hello world", "goodbye world"]))