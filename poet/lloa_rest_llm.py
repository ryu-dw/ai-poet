import requests
from typing import List, Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult


class WiseChatModel(BaseChatModel):
    api_url: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    model_name: str = "wisenut/wise-lloa-max-1.2.1"

    @property
    def _llm_type(self) -> str:
        return "wise_rest_chat_model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:

        # LangChain 메시지 → REST messages 변환
        payload_messages = []
        for m in messages:
            role = m.type
            # print(role)
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"

            payload_messages.append({"role": role, "content": m.content})

        payload = {
            "model": self.model_name,
            "messages": payload_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # REST 호출
        response = requests.post(
            self.api_url,
            json=payload,
            headers=headers,
            timeout=60,
        )

        response.raise_for_status()
        result = response.json()
        # print(f"API Response: {result}")

        # 응답 파싱 - 에러 처리 추가
        try:
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
            elif "error" in result:
                raise ValueError(f"API Error: {result['error']}")
            else:
                raise ValueError(f"Unexpected API response: {result}")

            if not content:
                raise ValueError("API returned empty content")

        except (KeyError, IndexError, TypeError) as e:
            print(f"API Response: {result}")
            raise ValueError(f"Failed to parse API response: {e}")

        ai_message = AIMessage(content=content)
        generation = ChatGeneration(message=ai_message)

        return ChatResult(generations=[generation])
