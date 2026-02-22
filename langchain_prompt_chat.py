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

prompt_text = chat_prompt_template.invoke({"history": history_data}).to_string()

model = ChatTongyi(model ="qwen-max")
res = model.invoke(input=prompt_text)
print(res.content,type(res))
