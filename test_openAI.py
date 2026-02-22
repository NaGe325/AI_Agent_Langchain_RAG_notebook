from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

#获取client实例，base_url参数请替换为阿里云百炼API的兼容模式地址
client = OpenAI(
    api_key=os.getenv("APIKEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# 调用接口
response = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=[{"role": "system", "content": "你是一个Ai助理。回答简介明了。"},
              {"role": "user", "content": "我有一只猫"},
              {"role": "assistant", "content": "好的，我明白了。"},
              {"role": "user", "content": "我还有一只狗。"},
              {"role": "assistant", "content": "好的，我知道了。"},
              {"role": "user", "content": "总共有几只宠物？"}
            ],
    stream=True
)
# 输出结果  
#print(response.choices[0].message.content)  

for chunk in response:
    content = chunk.choices[0].delta.content
    if content is not None:
        print(
            content,
            end=" ", # 每一段内容后添加一个空格，保持输出在同一行
            flush=True # 确保每次输出都立即显示，而不是等待缓冲区满了才显示
        )
print() # 输出完成后换行