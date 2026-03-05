import os
from pydoc import text
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Set
from langchain_core.documents import Document
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from io import StringIO
import tempfile
from lloa_rest_llm import WiseChatModel

# -----------------------------
# 1. 기본 UI
# -----------------------------
st.set_page_config(page_title="ChatPDF", layout="wide")
st.title("ChatPDF")
st.write("---")


# -----------------------------
# 2. LLM (전역 생성)
# -----------------------------
@st.cache_resource
def load_llm():
    LLOA_API_KEY = os.getenv("LLOA_API_KEY")
    return WiseChatModel(
        api_url="http://210.180.82.135:9023/v1/chat/completions",
        api_key=LLOA_API_KEY,
        temperature=0.0,
    )


load_dotenv()
#llm = load_llm()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# -----------------------------
# 3. PDF → Retriever 생성
# -----------------------------
@st.cache_resource
def build_retriever(uploaded_file):

    temp_dir = tempfile.TemporaryDirectory()
    temp_path = os.path.join(temp_dir.name, uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader(temp_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    texts = text_splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)

    vectorstore = Chroma.from_documents(texts, embeddings, collection_name="chatpdf")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    temp_dir.cleanup()

    return retriever


# -----------------------------
# 4. MultiQuery 생성 체인
# -----------------------------
query_prompt = ChatPromptTemplate.from_template(
    """
사용자 질문을 기반으로
의미적으로 서로 다른 검색 쿼리 3개를 생성하세요.
각 줄마다 하나씩 작성하세요.

질문: {question}
"""
)

query_chain = query_prompt | llm | StrOutputParser()


def generate_queries(question: str) -> List[str]:
    result = query_chain.invoke({"question": question})
    queries = [q.strip() for q in result.split("\n") if q.strip()]
    return queries


# -----------------------------
# 5. MultiQuery 검색
# -----------------------------
def multi_query_retrieve(question: str, retriever) -> List[Document]:

    queries = generate_queries(question)
    all_docs: List[Document] = []

    for q in queries:
        docs = retriever.invoke(q)
        all_docs.extend(docs)

    # 중복 제거
    unique_docs = list({doc.page_content: doc for doc in all_docs}.values())

    return unique_docs


# -----------------------------
# 6. 최종 QA 체인
# -----------------------------
answer_prompt = ChatPromptTemplate.from_template(
    """
아래 문서를 참고하여 질문에 답변하세요.

문서:
{context}

질문:
{question}
"""
)

answer_chain = answer_prompt | llm | StrOutputParser()


def ask(question: str, retriever):

    docs = multi_query_retrieve(question, retriever)
    context = "\n\n".join([d.page_content for d in docs])

    response = answer_chain.invoke({"context": context, "question": question})

    return response


# -----------------------------
# 7. Streamlit 실행 영역
# -----------------------------
uploaded_file = st.file_uploader("📂 PDF 파일을 업로드하세요", type=["pdf"])

if uploaded_file is not None:

    retriever = build_retriever(uploaded_file)

    st.success("✅ PDF 벡터 인덱싱 완료")

    question = st.text_input("💬 질문을 입력하세요")

    if question:
        with st.spinner("답변 생성 중..."):
            answer = ask(question, retriever)

        st.write("### 📌 답변")
        st.write(answer)
