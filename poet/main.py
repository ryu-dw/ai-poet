from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from lloa_rest_llm import WiseChatModel

load_dotenv()

# ChatOpenAI 초기화
llm = WiseChatModel(
    api_url="http://210.180.82.135:9023/v1/chat/completions",
    api_key="iBVxsB_eTwvEFOLnsEqBbD7Uo4ud31zgGXFKLgwppDM="
)

# response = chat_model.invoke([
#     SystemMessage(content="너는 친절한 상담원이다."),
#     HumanMessage(content="안녕? 좋은 아침이야")
# ])


# llm = init_chat_model("gpt-4o-mini", model_provider="openai")
# result = llm.invoke("hello")
# print(result)

# 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant"), ("user", "{input}")]
)

# 문자열 출력 파서
output_parser = StrOutputParser()

# LLM 체인 구성
chain = prompt | llm | output_parser
result = chain.invoke({"input": "hi"})
print(result)
