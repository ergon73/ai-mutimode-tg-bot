import logging
from typing import Any, Dict, List

import aiohttp
from openai import (
    APIConnectionError,
    APIError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessageParam

from config import settings

TOKEN_PRICES: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1-mini": {"input": 3.00, "output": 12.00},
}
FALLBACK_USD_RATE = 100.0


async def get_usd_rate() -> float:
    """
    Получить актуальный курс USD/RUB с сайта ЦБ РФ.

    Returns:
        Курс доллара к рублю; при ошибке возвращает запасное значение
    """
    url = "https://www.cbr-xml-daily.ru/daily_json.js"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json(content_type=None)
                    return float(data["Valute"]["USD"]["Value"])
                logging.warning(
                    "ЦБ РФ вернул неожиданный статус %s", response.status
                )
    except Exception as error:
        logging.warning("Не удалось получить курс ЦБ РФ: %s", error)
    return FALLBACK_USD_RATE


class LLMService:
    """
    Сервис взаимодействия с OpenAI API и расчёта стоимости ответов.
    """

    def __init__(self, api_key: str, model_name: str) -> None:
        """
        Создать экземпляр сервиса генерации ответов.

        Args:
            api_key: ключ OpenAI API
            model_name: имя используемой модели
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model_name = model_name

    async def generate_response(
        self, messages: List[ChatCompletionMessageParam]
    ) -> Dict[str, Any]:
        """
        Сгенерировать ответ модели на основе истории сообщений.

        Args:
            messages: список сообщений в формате OpenAI Chat API

        Returns:
            Словарь с текстом ответа и статистикой использования токенов
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
        except RateLimitError:
            logging.warning("Достигнут лимит запросов к OpenAI")
            raise
        except AuthenticationError:
            logging.error("Ошибка аутентификации OpenAI API")
            raise
        except APIConnectionError:
            logging.error("Проблемы подключения к OpenAI API")
            raise
        except APIError as error:
            logging.error("Ошибка OpenAI API: %s", error)
            raise

        usage = response.usage
        usage_data = self.calculate_cost(self.model_name, usage)
        response_text = response.choices[0].message.content if response.choices else ""

        return {
            "text": response_text,
            "usage": {
                "input_tokens": usage_data["input_tokens"],
                "output_tokens": usage_data["output_tokens"],
                "total_cost_usd": usage_data["total_cost_usd"],
            },
        }

    @staticmethod
    def calculate_cost(model_name: str, usage: Any) -> Dict[str, float]:
        """
        Рассчитать стоимость запроса на основе токенов и цен модели.

        Args:
            model_name: имя модели
            usage: объект usage из ответа OpenAI

        Returns:
            Информация о количестве токенов и стоимости запроса
        """
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        prices = TOKEN_PRICES.get(model_name, TOKEN_PRICES["gpt-4o-mini"])

        input_cost = (prompt_tokens / 1_000_000) * prices["input"]
        output_cost = (completion_tokens / 1_000_000) * prices["output"]
        total_cost_usd = input_cost + output_cost

        return {
            "input_tokens": float(prompt_tokens),
            "output_tokens": float(completion_tokens),
            "total_cost_usd": float(total_cost_usd),
        }

