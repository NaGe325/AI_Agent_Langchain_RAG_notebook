from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["DASHSCOPE_API_KEY"] = os.getenv("APIKEY")

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "写一首歌"),
        MessagesPlaceholder("history"),
        ("human","在写一首歌"),
    ]
)

history_data = [
    ("human","写一个歌"),
    ("ai","一闪一闪亮晶晶"),
    ("human","再来一首"),
    ("ai","夏天的风吹拂着大地，花儿在阳光下绽放。"),
]

model = ChatTongyi(model="qwen3-max")

#组成链，要求每一个组件都是Runnable接口的子类
chain = chat_prompt_template | model

#res = chain.invoke({"history": history_data})
#print(res.content)

for chunk in chain.stream({"history": history_data}):
    print(chunk.content, end="", flush=True)
    