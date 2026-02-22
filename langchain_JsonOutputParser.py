from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["DASHSCOPE_API_KEY"] = os.getenv("APIKEY")

str_parser = StrOutputParser()
json_parser = JsonOutputParser()  # 如果想要字符串输出，可以使用StrOutputParser
model = ChatTongyi(model="qwen3-max")

first_prompt = PromptTemplate.from_template(
    "我姓：{last_name},刚刚生了{gender},请帮我起名字，仅回复我名字,"
    "并封装成json格式。key是name,value是名字的字符串,严格遵守"
)

second_prompt = PromptTemplate.from_template(
    "姓名：{name},请帮我分析一下这个名字的含义？"
)

chain = first_prompt | model | json_parser | second_prompt | model | str_parser

res = chain.invoke({"last_name": "葛", "gender": "女"})
print(res)
print(type(res))

#可以stream输出
