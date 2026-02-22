from langchain_core.prompts import ChatPromptTemplate ,MessagesPlaceholder,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

from dotenv import load_dotenv
import os
load_dotenv()
os.environ["DASHSCOPE_API_KEY"] = os.getenv("APIKEY")

model = ChatTongyi(model="qwen3-max")
#prompt = PromptTemplate.from_template(
#    "你需要根据我提供的历史对话内容。会话历史:{chat_history}，用户的输入:{input}，请你根据这些内容来回答用户的问题。"
#)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你需要根据我提供的历史对话内容来回答用户的问题。"),
        MessagesPlaceholder("chat_history"),
        ("human", "用户的输入:{input}"),
    ]
)


str_parser = StrOutputParser()

def print_prompt(full_prompt):
    print("="*20,full_prompt.to_string(),"="*20)
    return full_prompt

base_chain = prompt | print_prompt | model | str_parser

store = {}
def get_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 定义一个新的链，继承RunnableWithMessageHistory
conversation_chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

if __name__ == "__main__":
    session_config = {
        "configurable": {
            "session_id": "test_session001"
        }
    }

    res = conversation_chain.invoke({"input": "我有一只猫"}, session_config)
    print(res)

    res = conversation_chain.invoke({"input": "我有一只狗"}, session_config)
    print(res)

    res = conversation_chain.invoke({"input": "我有多少宠物"}, session_config)
    print(res)