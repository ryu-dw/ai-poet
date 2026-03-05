import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
print(str(PROJECT_ROOT))

import os

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from poet.lloa_rest_llm import WiseChatModel
from dotenv import load_dotenv


def getLLMResponse(form_input, email_sender, email_recipient, language):
    """getLLMResponse 함수는 주어진 입력을 사용하여 LLM(대형 언어 모델)으로부터 이메일 응답을 생성합니다.

    매개변수:
    - form_input: 사용자가 입력한 이메일 주제.
    - email_sender: 이메일을 보낸 사람의 이름.
    - email_recipient: 이메일 받는 사람의 이름.
    - language: 이메일이 생성될 언어(한국어 또는 영어).

    반환값:
    - LLM이 생성한 이메일 응답 텍스트
    """

    load_dotenv()
    LLOA_API_KEY = os.getenv("LLOA_API_KEY")
    print("LLOA_API_KEY:", LLOA_API_KEY)

    llm = WiseChatModel(
        api_url="http://210.180.82.135:9023/v1/chat/completions", api_key=LLOA_API_KEY
    )

    # 프롬프트 템플릿 생성
    if language == "한국어":
        template = """
        {form_input} 주제를 포함한 이메일을 작성해줘. \n\n보낸 사람: {email_sender} \n받는 사람: {email_recipient} \n전부 {language}로 번역해서 작성해주세요. 한문은 내용에서 제외해주세요.
        \n\n이메일 내용:
    """
    else:
        template = """
        Write an email including the topic of {form_input}. \n\nSender: {email_sender} \nRecipient: {email_recipient} Please write the entire email in {language}. Exclude any Chinese characters from the content.
        \n\nEmail content:
        """

    prompt = PromptTemplate.from_template(template=template)
    # request = prompt.format(
    #     form_input=form_input,
    #     email_sender=email_sender,
    #     email_recipient=email_recipient,
    #     language=language,
    # )
    # print("LLM Request:", request)

    chain = prompt | llm | StrOutputParser()
    # LLM을 사용하여 응답 생성
    response = chain.invoke(
        {
            "form_input": form_input,
            "email_sender": email_sender,
            "email_recipient": email_recipient,
            "language": language,
        }
    )

    return response


st.set_page_config(
    page_title="이메일 생성기 📮",
    page_icon="📮",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.header("이메일 생성기 📮")
language_choice = st.selectbox(
    "이메일을 작성할 언어를 선택하세요", ("한국어", "English")
)
form_input = st.text_area("이메일 주제를 입력하세요", height=100)
col1, col2 = st.columns([10, 10])
with col1:
    email_sender = st.text_input("보낸 사람 이름")
with col2:
    email_recipient = st.text_input("받는 사람 이름")

submit = st.button("생성하기")

# '생성하기' 버튼이 클릭되면 LLM 응답을 가져와서 화면에 표시
if submit:
    with st.spinner("이메일을 생성하는 중입니다..."):
        response = getLLMResponse(
            form_input, email_sender, email_recipient, language_choice
        )
        st.subheader("생성된 이메일:")
        st.write(response)
