import os
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
# 프롬프트1 : 리뷰 번역
prompt1 = PromptTemplate.from_template(
    template="다음 숙박 시설 리뷰를 한글로 번역하세요.\n\n{review}"
)
chain1 = prompt1 | llm | StrOutputParser()

# 프롬프트2 : 리뷰 요약
prompt2 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰를 한 문장으로 요약하세요.\n\n{translation}"
)
chain2 = prompt2 | llm | StrOutputParser()

# 프롬프트3 : 번역된 리뷰 감성 점수 평가
prompt3 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰를 읽고 0점부터 10점 사이에서 긍정/부정 점수를 매기세요. 숫자만 대답하세요.\n\n{translation}"
)
chain3 = prompt3 | llm | StrOutputParser()

# 프롬프트4 : 원본 리뷰 언어 식별
prompt4 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰의 원본 언어를 식별하세요. 언어 이름만 대답하세요.\n\n{review}"
)
chain4 = prompt4 | llm | StrOutputParser()

# 프롬프트5 : 요약에 대한 공손한 답변 생성(원본 언어 사용)
prompt5 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰 요약에 대해 공손한 답변을 작성하세요.\n답변 언어:{language}\n리뷰 요약:{summary}"
)
chain5 = prompt5 | llm | StrOutputParser()

# 프롬프트6 : 생성된 답변을 한국어로 번역
prompt6 = PromptTemplate.from_template(
    "다음 숙박 시설 리뷰에 대한 답변을 한국어로 번역하세요.\n\n{reply}"
)
chain6 = prompt6 | llm | StrOutputParser()

# RunnablePassthrough.assign을 사용하여 각 단계의 출력을 다음 단계의 입력으로 전달하고, 중간 결과들을 딕셔너리에 누적합니다.
all_chain = (
    RunnablePassthrough.assign(
        translation=chain1
    )
    .assign(
        summary=chain2
    )
    .assign(
        sentiment_score=chain3
    )
    .assign(
        language=chain4
    )
    .assign(
        reply=chain5
    )
    .assign(
        korean_reply=chain6
    )
)
review = """This hotel has a great location and the rooms are clean. 
However, the staff were not very friendly and the breakfast was disappointing. 
Overall, I had a decent stay but there is room for improvement."""

try:
    result = all_chain.invoke({"review": review})
    print(f'translation 결과 \n{result["translation"]}')
    print(f'summary 결과 \n{result["summary"]}')
    print(f'sentiment_score 결과 \n{result["sentiment_score"]}')
    print(f'language 결과 \n{result["language"]}')
    print(f'reply 결과 \n{result["reply"]}')
    print(f'korean_reply 결과 \n{result["korean_reply"]}')
except Exception as e:
    print(f"Error: {e}")