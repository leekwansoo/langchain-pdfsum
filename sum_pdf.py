import streamlit as st 
import os 
from PyPDF2 import PdfReader
from openai import OpenAI


from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

# API Key 설정
#open_api_key= os.getenv("OPENAI_API_KEY")
open_api_key= st.secrets["OPENAI_API_KEY"]
print(open_api_key)
os.environ["OPENAI_API_KEY"] = open_api_key

from utils import print_messages
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

st.title("PDF 요약하기")
st.divider()

pdf = st.file_uploader("PDF File_Upload", type= 'pdf')

if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    text_splitter = CharacterTextSplitter(
        separator= "\n", 
        chunk_size = 1000,
        chunk_overlap = 200, 
        length_function = len
    )
    
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    documents = FAISS.from_texts(chunks, embeddings)
    
    query = " 업로드된 PDF 파일의 내용을 10 문장 정도로 요약해 주세요."
    
    docs = documents.similarity_search(query)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.1)
    chain = load_qa_chain(llm, chain_type="stuff")
    
    with get_openai_callback() as cost:
        response = chain.run(input_documents=docs, question = query)
        print(cost)
    
    st.subheader("요약결과")
    st.write(response)
