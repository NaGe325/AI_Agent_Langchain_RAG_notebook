from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate
from langchain_community.llms import Tongyi
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["DASHSCOPE_API_KEY"] = os.getenv("APIKEY")

example_template = PromptTemplate.from_template("Question: {question}\nAnswer: {answer}")

example_data = [
    {"question": "What is the capital of France?", "answer": "Paris"},
]
few_shot_template = FewShotPromptTemplate(
    example_prompt=example_template,
    examples=example_data,
    prefix="Tell me the answer to the capital of countries,I will give you the example:",
    suffix="Based on the above example, answer the following question:\n{input_words}",
    input_variables=['input_words']
)

prompt_text = few_shot_template.invoke(input={"input_words": "What is the capital of China?"}).to_string()
print(prompt_text)

model = Tongyi(model ="qwen-max")
print(model.invoke(input=prompt_text))