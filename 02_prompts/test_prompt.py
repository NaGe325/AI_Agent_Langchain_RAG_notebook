from openai import OpenAI
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

examples_data = {}

examples_types = {}

questions = []
"""
[
    {"role": "system", "content": "xxx"},
    {"role": "user", "content": "我有一只猫"},
    {"role": "assistant", "content": "好的，我明白了。"},
]
"""

messages = [
    {"role" : "system", "content": "xxxxx"},
]


for key, value in examples_data.items():
    messages.append({"role": "user", "content":value})
    messages.append({"role": "assistant", "content": key})


#向模型提问
for q in questions:
    response = client.chat.completions.create(
        model = "qwen3.5-plus",
        messages = messages + [{"role": "user", "content": q}],
    )
    
    print(response.choices[0].message.content)
