from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["DASHSCOPE_API_KEY"] = os.getenv("APIKEY")

parser = StrOutputParser()
model = ChatTongyi(model="qwen3-max")

prompt = PromptTemplate.from_template(
    "What is the capital of {country}?"
)

chain = prompt | model | parser | model

res = chain.invoke({"country": "France"})
print(res.content)
