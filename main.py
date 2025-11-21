import asyncio
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError

from config import settings
from llm_service import LLMService, get_usd_rate
from memory import MemoryManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_FILE = BASE_DIR / "prompts.json"

WELCOME_MESSAGE = """
👋 Привет! Я AI-ассистент с несколькими режимами работы.

📚 Доступные команды:
/mode — выбрать режим работы
/reset — очистить историю диалога
/start — показать это сообщение

Текущий режим: {current_mode}
"""

COST_TEMPLATE = """
💰 Стоимость запроса:
📥 Входные токены: {input_tokens}
📤 Выходные токены: {output_tokens}
💵 USD: ${cost_usd:.5f}
💸 RUB: ~{cost_rub:.2f}₽
"""

TELEGRAM_MESSAGE_LIMIT = 4096

router = Router()
bot = Bot(token=settings.BOT_TOKEN)
dp = Dispatcher()
dp.include_router(router)

llm_service = LLMService(
    api_key=settings.OPENAI_API_KEY,
    model_name=settings.OPENAI_MODEL,
)


@lru_cache
def load_prompts() -> Dict[str, Any]:
    """
    Загрузить описание ролей из файла prompts.json.

    Returns:
        Словарь с настройками ролей и значением режима по умолчанию
    """
    with PROMPTS_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def get_default_mode() -> str:
    """
    Получить режим по умолчанию из файла с подсказками.

    Returns:
        Ключ режима по умолчанию
    """
    prompts_data = load_prompts()
    return prompts_data.get("default_prompt", "assistant")


memory_manager = MemoryManager(
    limit=settings.MAX_HISTORY_MESSAGES,
    default_mode=get_default_mode(),
)


def split_into_chunks(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> List[str]:
    """
    Разбить длинный текст на части, укладывающиеся в лимит Telegram.

    Args:
        text: исходная строка
        limit: максимальная длина части

    Returns:
        Список строк, подходящих для отправки
    """
    normalized = text.strip()
    if not normalized:
        return []

    chunks: List[str] = []
    start = 0
    length = len(normalized)

    while start < length:
        end = min(start + limit, length)
        chunk = normalized[start:end]

        if end < length:
            split_pos = chunk.rfind("\n")
            if split_pos == -1:
                split_pos = chunk.rfind(" ")
            if split_pos > 0:
                end = start + split_pos
                chunk = normalized[start:end]

        chunks.append(chunk.strip())
        start = end

    return [part for part in chunks if part]


def get_mode_keyboard() -> InlineKeyboardMarkup:
    """
    Сформировать клавиатуру для переключения режимов бота.

    Returns:
        Объект InlineKeyboardMarkup с кнопками режимов
    """
    prompts_data = load_prompts()
    keyboard_buttons: List[List[InlineKeyboardButton]] = []
    for key, prompt in prompts_data["prompts"].items():
        keyboard_buttons.append(
            [
                InlineKeyboardButton(
                    text=prompt["name"],
                    callback_data=f"mode:{key}",
                )
            ]
        )
    return InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)


def get_system_prompt(mode: str) -> str:
    """
    Получить system prompt для выбранного режима.

    Args:
        mode: ключ режима

    Returns:
        Строка system_prompt
    """
    prompts_data = load_prompts()
    return prompts_data["prompts"].get(mode, prompts_data["prompts"]["assistant"])["system_prompt"]


@router.message(Command("start"))
async def handle_start(message: Message) -> None:
    """
    Обработать команду /start и отправить приветственное сообщение.
    """
    current_mode = memory_manager.get_mode(message.chat.id)
    await message.answer(WELCOME_MESSAGE.strip().format(current_mode=current_mode))


@router.message(Command("mode"))
async def handle_mode(message: Message) -> None:
    """
    Показать пользователю клавиатуру с доступными режимами.
    """
    await message.answer("Выберите режим работы:", reply_markup=get_mode_keyboard())


@router.callback_query(F.data.startswith("mode:"))
async def handle_mode_callback(callback: CallbackQuery) -> None:
    """
    Обработать переключение режима через inline-кнопку.
    """
    mode_key = callback.data.split("mode:", maxsplit=1)[1]
    prompts_data = load_prompts()
    if mode_key not in prompts_data["prompts"]:
        await callback.answer("❌ Неизвестный режим", show_alert=True)
        return
    memory_manager.set_mode(callback.message.chat.id, mode_key)
    await callback.answer(f"✅ Режим переключён: {prompts_data['prompts'][mode_key]['name']}")
    await callback.message.answer(f"Текущий режим: {prompts_data['prompts'][mode_key]['name']}")


@router.message(Command("reset"))
async def handle_reset(message: Message) -> None:
    """
    Очистить историю сообщений для текущего чата.
    """
    memory_manager.clear_history(message.chat.id)
    await message.answer("✅ Память диалога очищена")


@router.message(F.text)
async def handle_text_message(message: Message) -> None:
    """
    Обработать текстовое сообщение пользователя и вернуть ответ модели.
    """
    mode = memory_manager.get_mode(message.chat.id)
    system_prompt = get_system_prompt(mode)
    history = memory_manager.get_history(message.chat.id)
    context_messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": message.text},
    ]

    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    try:
        llm_result = await llm_service.generate_response(context_messages)
    except RateLimitError:
        await message.answer("⏱ Слишком много запросов. Подождите немного.")
        return
    except AuthenticationError:
        await message.answer("❌ Ошибка аутентификации API")
        return
    except APIConnectionError:
        await message.answer("🌐 Проблемы с подключением. Попробуйте позже.")
        return
    except APIError:
        await message.answer("❌ Ошибка сервера. Попробуйте позже.")
        return
    except Exception as error:  # pylint: disable=broad-except
        logger.error("Неожиданная ошибка генерации: %s", error)
        await message.answer("❌ Произошла ошибка. Попробуйте позже.")
        return

    memory_manager.add_message(message.chat.id, "user", message.text)
    memory_manager.add_message(message.chat.id, "assistant", llm_result["text"])

    usage = llm_result["usage"]
    usd_rate = await get_usd_rate()
    cost_rub = usage["total_cost_usd"] * usd_rate
    response_chunks = split_into_chunks(llm_result["text"])
    if not response_chunks:
        response_chunks = ["🤖 Модель не вернула текст ответа."]

    for chunk in response_chunks:
        await message.answer(chunk)

    cost_text = COST_TEMPLATE.strip().format(
        input_tokens=int(usage["input_tokens"]),
        output_tokens=int(usage["output_tokens"]),
        cost_usd=usage["total_cost_usd"],
        cost_rub=cost_rub,
    )
    await message.answer(cost_text)


async def main() -> None:
    """
    Точка входа для запуска Telegram-бота.
    """
    prompts_data = load_prompts()
    default_mode = prompts_data.get("default_prompt", "assistant")
    logger.info("Загружено ролей: %s", len(prompts_data.get("prompts", {})))
    logger.info("Модель OpenAI: %s", settings.OPENAI_MODEL)
    logger.info("Режим по умолчанию: %s", default_mode)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен пользователем")
