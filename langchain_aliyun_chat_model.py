from langchain_community.chat_models.tongyi import ChatTongyi
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
load_dotenv()
os.environ["DASHSCOPE_API_KEY"] = os.getenv("APIKEY")

model=ChatTongyi(model="qwen-max")

'''
messages=[
    SystemMessage(content="你是留学生，喜欢写短小的歌"),
    HumanMessage(content="写一首短小的歌"),
    AIMessage(content="孤独是夜空中最亮的星，照亮我前行的路"),
    HumanMessage(content="再写一首"),
]
'''
# 上面是使用了 langchain_core 定义的消息格式，下面是直接使用元组列表的方式
messages=[
    #(role, content)
    ("system","你是留学生，喜欢写短小的歌"),
    ("human","写一首短小的歌"),
    ("ai","孤独是夜空中最亮的星，照亮我前行的路"),
    ("human","再写一首"),
]


res = model.stream(input=messages)

for chunk in res:
    print(chunk.content, end="", flush=True)