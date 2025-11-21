import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_MODE = "assistant"


class MemoryManager:
    """
    Класс для управления историей сообщений и режимами пользователей.
    """

    def __init__(
        self,
        storage_path: Path | str = "memory.json",
        limit: int = 10,
        default_mode: str = DEFAULT_MODE,
    ) -> None:
        """
        Инициализировать менеджер памяти и загрузить данные с диска.

        Args:
            storage_path: путь до файла хранения памяти
            limit: максимальное количество сообщений в истории
        """
        self.storage_path = Path(storage_path)
        self.limit = limit
        self.default_mode = default_mode
        self._data: Dict[str, Dict[str, Any]] = {}
        self.load()

    def add_message(self, chat_id: int, role: str, content: str) -> None:
        """
        Добавить сообщение в историю конкретного чата.

        Args:
            chat_id: идентификатор чата
            role: роль автора сообщения (system/user/assistant)
            content: текст сообщения
        """
        chat_key = str(chat_id)
        chat_data = self._data.setdefault(
            chat_key, {"history": [], "mode": self.default_mode}
        )
        chat_data["history"].append({"role": role, "content": content})
        chat_data["history"] = chat_data["history"][-self.limit :]
        self.save()

    def get_history(self, chat_id: int, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Получить историю сообщений для выбранного чата.

        Args:
            chat_id: идентификатор чата
            limit: ограничение количества возвращаемых сообщений

        Returns:
            Список сообщений для формирования контекста
        """
        chat_key = str(chat_id)
        history = self._data.get(chat_key, {}).get("history", [])
        if limit is None:
            return history.copy()
        return history[-limit:].copy()

    def clear_history(self, chat_id: int) -> None:
        """
        Очистить историю сообщений для выбранного чата.

        Args:
            chat_id: идентификатор чата
        """
        chat_key = str(chat_id)
        if chat_key in self._data:
            self._data[chat_key]["history"] = []
            self.save()

    def set_mode(self, chat_id: int, mode: str) -> None:
        """
        Сохранить выбранный режим для чата.

        Args:
            chat_id: идентификатор чата
            mode: ключ режима
        """
        chat_key = str(chat_id)
        chat_data = self._data.setdefault(
            chat_key, {"history": [], "mode": self.default_mode}
        )
        chat_data["mode"] = mode
        self.save()

    def get_mode(self, chat_id: int) -> str:
        """
        Получить текущий режим для чата.

        Args:
            chat_id: идентификатор чата

        Returns:
            Ключ активного режима
        """
        chat_key = str(chat_id)
        return self._data.get(chat_key, {}).get("mode", self.default_mode)

    def save(self) -> None:
        """
        Сохранить текущее состояние памяти в файл.
        """
        try:
            self.storage_path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as error:
            logging.error(f"Не удалось сохранить память: {error}")

    def load(self) -> None:
        """
        Загрузить состояние памяти из файла, если он существует.
        """
        if not self.storage_path.exists():
            return
        try:
            raw_data = self.storage_path.read_text(encoding="utf-8")
            self._data = json.loads(raw_data)
        except (OSError, json.JSONDecodeError) as error:
            logging.error(f"Не удалось загрузить память: {error}")
            self._data = {}

