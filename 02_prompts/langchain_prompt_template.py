from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["DASHSCOPE_API_KEY"] = os.getenv("APIKEY")

prompt_template = PromptTemplate.from_template("What is the capital of {country}?")


#prompt_text = prompt_template.format(country="France")
model = Tongyi(model="qwen-max")
#res = model.invoke(prompt_text)
#print(res)  # 输出: What is the capital of France?

chain = prompt_template | model
res= chain.invoke(input={"country": "France"})
print(res)



