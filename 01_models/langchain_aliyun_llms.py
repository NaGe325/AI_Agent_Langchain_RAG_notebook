#langchain_community
from langchain_community.llms.tongyi import Tongyi
import os
from dotenv import load_dotenv

load_dotenv()

# 将你的 APIKEY 赋值给 LangChain 和 DashScope 期望的环境变量名
os.environ["DASHSCOPE_API_KEY"] = os.getenv("APIKEY")

# 获取模型实例，现在不需要显式传 dashscope_api_key 了，它会自动去读环境变量
model = Tongyi(
    model="qwen-max"
)

#调用invoke方法，输入参数请替换为您想要测试的内容
#调用stream方法，输入参数请替换为您想要测试的内容
res = model.stream(
    input="who are you?",
)

for chunk in res:
        print(chunk, end="", flush=True)
