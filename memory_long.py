import os,json
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import message_to_dict,messages_from_dict,BaseMessage
from langchain_core.prompts import ChatPromptTemplate ,MessagesPlaceholder,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory

from dotenv import load_dotenv
import os
load_dotenv()
os.environ["DASHSCOPE_API_KEY"] = os.getenv("APIKEY")
class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id,file_path):
        self.session_id = session_id
        self.file_path = file_path
        #total_path
        self.file_path = os.path.join(self.file_path,self.session_id)
        #create the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.file_path),exist_ok=True)

    def add_messages(self,messages:Sequence[BaseMessage]) -> None:
        
        all_messages = list(self.messages) #先获取当前的消息记录
        all_messages.extend(messages) #将新的消息添加到当前的消息记录中
        #将所有的消息记录转换为字典格式，并写入文件中，官方的message_to_dict函数转换单条消息成为字典格式
        new_messages = [message_to_dict(m) for m in all_messages] #将新的消息转换为字典格式
        #将新的消息记录写入文件中，官方的json.dump函数可以将python对象转换为json格式，并写入文件中
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump(new_messages,f) #将新的消息记录写入文件中

    @property #定义一个属性方法，获取当前的消息记录
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path,"r",encoding="utf-8") as f:
                messages_data = json.load(f) #从文件中读取消息记录，并转换为字典格式
                return messages_from_dict(messages_data) #将字典格式的消息记录转换为BaseMessage对象，并返回
        except FileNotFoundError:
            return [] #如果文件不存在，返回一个空列表
        
    def clear(self) -> None:
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump([],f) #将一个空列表写入文件中，清空消息记录   



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


def get_history(session_id):
    #return InMemoryChatMessageHistory() #每次都返回一个新的InMemoryChatMessageHistory对象，无法保存历史消息记录
    return FileChatMessageHistory(session_id,"./chat_history") #返回一个FileChatMessageHistory对象，可以保存历史消息记录

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