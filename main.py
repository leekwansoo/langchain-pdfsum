import streamlit as st 
import os 
from openai import OpenAI

open_api_key= os.getenv("OPENAI_API_KEY")
#print(open_api_key)

os.environ["OPENAI_API_KEY"] = open_api_key
llm = OpenAI()

st.set_page_config(page_title="ChatGPT", page_icon="")
st.title("ChatGPT")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 이전 대화 기록 프린트  
def print_messages():  
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        for role, message in st.session_state["messages"]:
            st.chat_message(role).write(message)

print_messages()

if user_input := st.chat_input("Enter your message"):
    st.chat_message("user").write(f'{user_input}')
    st.session_state["messages"].append(("user", user_input)) # append as tuple
    
    with st.chat_message("assistant"):
        msg = f"당신이 입력한 내용: {user_input}"
        st.write(msg)
        st.session_state["messages"].append(("assistant", msg)) # append as tuple
