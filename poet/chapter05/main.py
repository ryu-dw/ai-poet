import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
print(str(PROJECT_ROOT))

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from poet.lloa_rest_llm import WiseChatModel
import streamlit as st

#load_dotenv()
LLOA_API_KEY = os.getenv("LLOA_API_KEY")

llm = WiseChatModel(
    api_url="http://210.180.82.135:9023/v1/chat/completions", api_key=LLOA_API_KEY
)

# 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant"), ("user", "{input}")]
)

# 문자열 출력 파서
output_parser = StrOutputParser()

# LLM 체인 구성
chain = prompt | llm | output_parser

st.title("인공지능 시인")
content = st.text_input("시의 주제를 입력하세요")

#시 작성 요청하기
if st.button("시 작성 요청하기"):
    with st.spinner("Wait for it..."):
        response = chain.invoke({"input": content + "에 대한 시를 작성해줘"})
        st.write(response)
