from cryptography.fernet import Fernet

from config import settings


class SecretEncryptor:
    """Шифрование секретов (например, креды job-сайтов) с помощью Fernet.

    Если FERNET_KEY не задан, шифрование отключено (значения хранятся как есть) —
    это режим локальной разработки; в продакшене ключ обязателен.
    """

    def __init__(self) -> None:
        key = settings.FERNET_KEY.encode()
        self._fernet = Fernet(key) if key else None

    def encrypt(self, value: str) -> str:
        """Зашифровать строку. Пустые значения и отсутствие ключа — без изменений.

        :param value: исходная строка
        :return: зашифрованная строка (base64) или исходная, если шифрование выключено
        """
        if not value or self._fernet is None:
            return value
        return self._fernet.encrypt(value.encode()).decode()

    def decrypt(self, value: str) -> str:
        """Расшифровать строку, зашифрованную encrypt().

        :param value: зашифрованная строка
        :return: исходная строка (или value, если шифрование выключено)
        """
        if not value or self._fernet is None:
            return value
        return self._fernet.decrypt(value.encode()).decode()


encryptor = SecretEncryptor()
