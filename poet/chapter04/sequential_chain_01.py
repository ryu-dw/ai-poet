import sys
import os
from unittest import result
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from ..lloa_rest_llm import WiseChatModel  # 부모 디렉토리에서 import
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
LLOA_API_KEY = os.getenv("LLOA_API_KEY")

llm = WiseChatModel(
    api_url="http://210.180.82.135:9023/v1/chat/completions", api_key=LLOA_API_KEY
)

prompt1 = PromptTemplate.from_template(
    "다음 식당 리뷰를 한 문장으로 요약하세요.\n\n{review}"
)
chain1 = prompt1 | llm | StrOutputParser()

prompt2 = PromptTemplate.from_template(
    "다음 식당 리뷰를 읽고 0점부터 10점 사이에서 긍정/부정 점수를 매기세요. 숫자만 대답하세요.\n\n{review}"
)
chain2 = prompt2 | llm | StrOutputParser()

prompt3 = PromptTemplate.from_template(
    "다음 식당 리뷰 요약에 대해 공손한 답변을 작성하세요.\n리뷰 요약:{summary}"
)
chain3 = prompt3 | llm | StrOutputParser()

# | 기호를 사용하여 프롬프트와 llm, 출력 파서를 연결할 수 있습니다.
all_chain = (
    RunnablePassthrough.assign(
        summary=chain1
    )
    .assign(
        sentiment_score=chain2
    )
    .assign(
        reply=chain3
    )
)
review = """이 식당은 맛도 좋고 분위기도 좋았습니다. 가격 대비 만족도가 높아요.
하지만, 서비스 속도가 너무 느려서 조금 실망스러웠습니다. 전반적으로는 다시 방문할 의향이 있습니다."""
try:
    result = all_chain.invoke({"review": review})
    print(f'summary 결과 \n{result["summary"]}')
    print(f'sentiment_score 결과 \n{result["sentiment_score"]}')
    print(f'reply 결과 \n{result["reply"]}')
except Exception as e:
    print(f"Error: {e}")