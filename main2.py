import streamlit as st 
import os 
from openai import OpenAI

# API Key 설정
#open_api_key= os.getenv("OPENAI_API_KEY")
open_api_key= st.secrets["OPENAI_API_KEY"]
#print(open_api_key)
os.environ["OPENAI_API_KEY"] = open_api_key

from utils import print_messages
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="ChatGPT", page_icon="")
st.title("ChatGPT")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 이전 대화 기록 프린트    
print_messages()

if user_input := st.chat_input("Enter your message"):
    # 사용자가 입력한 내용
    st.chat_message("user").write(f'{user_input}')
    #st.session_state["messages"].append(("user", user_input)) # append as tuple
    st.session_state["messages"].append(ChatMessage(role = "user", content = user_input)) 
    
    
    # LLM을 사용하여 AI 답변 생성
    prompt = ChatPromptTemplate.from_template(
        """질문에 대하여 간결하게 답변해 주세요
        {question}
        """)
    
    chain = prompt | ChatOpenAI() | StrOutputParser()
    #response = chain.invoke({"question": user_input}) 
    #msg = response.content
    msg = chain.invoke({"question": user_input}) # pipelining
    
    # AI의 답변
    with st.chat_message("assistant"):
        # msg = f"당신이 입력한 내용: {user_input}"
        st.write(msg)
        st.session_state["messages"].append(ChatMessage(role = "assistant", content = msg)) # append as tuple
