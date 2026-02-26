import sys
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from ..lloa_rest_llm import WiseChatModel  # 부모 디렉토리에서 import
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
LLOA_API_KEY = os.getenv("LLOA_API_KEY")

prompt_template = " 이 음식의 리뷰 '{review}'에 대해 '{rating1}'점부터 '{rating2}'점까지의 평가를 해주세요"
prompt = PromptTemplate(
    input_variables=["review", "rating1", "rating2"], template=prompt_template
)

llm = WiseChatModel(
    api_url="http://210.180.82.135:9023/v1/chat/completions", api_key=LLOA_API_KEY
)

# | 기호를 사용하여 프롬프트와 llm, 출력 파서를 연결할 수 있습니다.
chain = prompt | llm | StrOutputParser()

try:
    response = chain.invoke({"review": "맛있어요", "rating1": "1", "rating2": "5"})
    print(f"평가 결과:{response}")
except Exception as e:
    print(f"Error: {e}")
