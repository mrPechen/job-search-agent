# Мульти-сайтовый поиск вакансий через VLM + LLM

Дата: 2026-08-25

## Цель

Сейчас агент ищет вакансии только на hh.ru: доменный whitelist в browser-сервисе
жёстко ограничен `{"hh.ru", "www.hh.ru", "localhost", "127.0.0.1"}`, а разбор
страницы завязан на hh-специфичные CSS-селекторы (`HhAdapter`, `HhSearcher`,
`submit_application`).

Цель — поддержка **произвольных** сайтов по работе. Пользователь при первом
общении указывает, на каких сайтах искать; агент запоминает список и ищет на них
под аккаунтом пользователя, разбирая страницы универсально через VLM (vision) +
LLM и управляя браузером через Playwright. После нахождения вакансий агент
оценивает релевантность, пишет сопроводительное письмо и откликается (с
подтверждением человека).

Единый путь разбора для всех сайтов, включая hh.ru — без детерминированных
адаптеров. Работа как с локальными моделями (Ollama), так и с облачными API
(OpenAI).

## Требования

1. Onboarding: при первом контакте бот спрашивает, на каких сайтах искать, и
   сохраняет список доменов для пользователя.
2. Поиск выполняется на всех сохранённых сайтах под аккаунтом пользователя
   (режим CDP — его Chrome с логинами, либо персистентный профиль с входом).
3. Разбор структуры сайта (поиск, результаты, кнопка отклика) — универсально
   через VLM + LLM, без хардкода селекторов под конкретный сайт.
4. Авто-отклик: найденные вакансии оцениваются по профилю, для подходящих пишется
   письмо и выполняется отклик с HITL-подтверждением.
5. Провайдер LLM/VLM выбирается конфигом: `ollama` (локально) или `openai`
   (облако). Оба пути должны работать одинаково.

## Архитектура

Новые/изменённые компоненты:

- `src/agent/browser_loop.py` (новый) — браузерный цикл с VLM.
- `src/agent/searcher.py` (новый) — `UniversalSearcher` и `UniversalApplier`.
- `src/browser/server.py` — новые экшены `scroll`, `back`; `navigate` принимает
  `allowed_domains`.
- `src/browser/executor.py` — методы `scroll`, `back`, `navigate(..., allowed_domains)`.
- `src/database/models.py` — новая таблица `user_sites`; `User.status` для onboarding.
- `src/tg/service.py` — onboarding-логика до запуска графа.
- `src/tg/bot.py` — композиционный корень: новые `searcher`/`applier`.
- `config.py` — `BROWSER_MAX_STEPS`.

Удаляемые/неиспользуемые компоненты:

- `src/browser/searcher.py` (`HhSearcher`) — заменяется `UniversalSearcher`.
- hh-специфичный `submit_application` в `src/browser/server.py` и
  `BrowserExecutor.submit_application` — заменяется apply-циклом.
- `src/browser/adapters.py` (`HhAdapter`, `GenericAdapter`) — селекторы больше не нужны.

Граф агента (`src/agent/graph.py`) не меняется структурно: меняются только
реализации `searcher` и `applier`, передаваемые в `build_graph`.

## Модель данных

### `user_sites`

```python
class UserSite(Base):
    __tablename__ = "user_sites"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    domain: Mapped[str] = mapped_column(String(255))
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
```

Unique constraint `(user_id, domain)`. `domain` хранится нормализованным
(только hostname без схемы и пути, например `habr.com`).

### `User.status`

Используется для onboarding: при создании пользователя — `"onboarding"`, после
сохранения хотя бы одного сайта — `"active"`. Существующие пользователи (уже
`"active"`) не затрагиваются; onboarding срабатывает только для новых.

## Onboarding

В `TgService.handle_message`, до вызова графа:

1. Получить/создать пользователя.
2. Если у пользователя нет сайтов в `user_sites`:
   - парсим сообщение на домены: сначала LLM `invoke_structured` со схемой
     `Sites{domains: list[str]}`, при неудаче — regex-фолбэк `[a-z0-9.-]+\.[a-z]{2,}`;
   - если домены найдены — сохранить их, перевести статус в `"active"`, ответить
     списком сохранённых сайтов;
   - если не найдены — ответить «На каких сайтах искать работу? Назови через
     запятую» и не запускать граф.
3. Если сайты есть — запустить граф как обычно.

Onboarding не трогает граф и роутер: это отдельный pre-graph шаг.

## Динамический whitelist доменов

`NavigateRequest` получает поле `allowed_domains: list[str] | None`. Проверка
`_is_allowed_url(url, allowed_domains)`:

- hostname входит в `allowed_domains` (точное совпадение или поддомен) И
- схема — `http` или `https`.

Если `allowed_domains` не передан — используется статический `ALLOWED_DOMAINS`
(обратная совместимость и тесты). Агент всегда передаёт сайты конкретного
пользователя.

## Браузерный цикл с VLM

Модуль `src/agent/browser_loop.py`. Схемы:

```python
class BrowserAction(BaseModel):
    tool: Literal["navigate", "click", "type", "scroll", "back", "done"]
    args: dict

class SearchOutcome(BaseModel):
    candidates: list[dict]  # {title, url, description, site}

class ApplyOutcome(BaseModel):
    applied: bool
    detail: str
```

Цикл параметризуется целью и схемой результата (`SearchOutcome` или `ApplyOutcome`).
`BrowserLoop.run(goal, result_schema, max_steps=settings.BROWSER_MAX_STEPS)`:

1. Получить состояние страницы: `extract()` (видимый текст + интерактивные
   элементы с CSS-селекторами) и `screenshot()` (PNG).
2. Собрать VLM-сообщение: текстовая инструкция с целью + список элементов с
   селекторами + скриншот как `image_url` (base64 data-URL).
3. Вызвать `gateway.invoke_structured(gateway.vision_model, messages, BrowserAction)`.
4. Если `tool == "done"` — провалидировать `args` против `result_schema` и вернуть
   результат.
5. Иначе выполнить действие через `BrowserExecutor` (`navigate`/`click`/`type`/
   `scroll`/`back`) и повторить.
6. Лимит шагов исчерпан — вернуть частичный результат / `applied=False`.

Две цели (goal-строки):

- **search** — «найди поле поиска, введи запрос `<query>`, получи список вакансий
  (title + url + короткое описание) со страницы результатов».
- **apply** — «открой вакансию `<url>`, нажми кнопку отклика, впиши сопроводительное
  письмо, отправь и подтверди успех».

Элементы отдаются текстом вместе со скриншотом, чтобы модель выбирала селектор из
реального DOM, а не угадывала его по картинке.

## Интеграция с графом

### `UniversalSearcher`

Интерфейс прежний: `async __call__(user_id, query) -> list[dict]`.

1. Загрузить сайты пользователя из `user_sites`.
2. Определить запрос поиска: `profile.desired_role`, если заполнен; иначе `query`.
3. Для каждого сайта запустить `BrowserLoop` в режиме search (стартовый
   `navigate` на `https://{domain}`) и собрать кандидатов со всех сайтов.

### `UniversalApplier`

Интерфейс прежний: `async __call__(user_id, decision) -> None`.

Для `decision["job"]["url"]` запускает `BrowserLoop` в режиме apply, передавая
`decision["cover_letter"]`. Пишет результат в аудит (`log_action`) как сейчас.

### Композиционный корень

В `src/tg/bot.py::build_tg_service` вместо `HhSearcher`/`submit_application`
собираются `UniversalSearcher` и `UniversalApplier` (зависят от `BrowserExecutor`,
`LLMGateway`, session factory). `send_message` (переписка с работодателем) остаётся
незатронутым.

## Browser-сервис

- `POST /scroll` (`{user_id, delta}`) — прокрутка страницы.
- `POST /back` (`{user_id}`) — `page.go_back()`.
- `POST /navigate` — `{user_id, url, allowed_domains}`.
- `POST /submit` — удаляется (заменяется apply-циклом).

`BrowserExecutor` добавляет соответствующие методы.

## Конфигурация

```python
LLM_PROVIDER: str = "openai"        # openai | ollama
LLM_TEXT_MODEL: str = "gpt-4o-mini"
LLM_VISION_MODEL: str = "gpt-4o"    # облако: gpt-4o; ollama: qwen3-vl
BROWSER_MAX_STEPS: int = 15
```

`LLM_VISION_MODEL` начинает реально использоваться циклом. Оба провайдера
(`openai`/`ollama`) строятся через `build_vision_model` и отдают
`langchain BaseChatModel`, поэтому цикл не зависит от провайдера.

## Обработка ошибок

- Лимит шагов исчерпан → частичный результат (search — то, что успели собрать;
  apply — `applied=False`) + лог.
- VLM не распарсила `BrowserAction` → 1 ретрай, затем для search — `[]`,
  для apply — `applied=False` с пояснением.
- Домен не в whitelist → 403 (существующее поведение).
- Селектор не найден (`LookupError`) → цикл получает прежнюю страницу и пробует
  снова (VLM видит отсутствие изменений и скорректирует действие).

## Безопасность

- HITL-подтверждение для `apply` сохраняется (`requires_human_approval`).
- Гардрейл `check_outgoing_message` для исходящих сообщений сохраняется.
- Навигация ограничена сайтами конкретного пользователя + только `http/https`.

## Тестирование

- Юнит: парсинг доменов из текста (LLM-фолбэк regex), матчинг whitelist
  (точное/поддомен), схема `BrowserAction`, разрешение запроса поиска
  (`desired_role` vs `query`), onboarding-флоу.
- Цикл на моках: `FakeGateway` (возвращает заданные `BrowserAction`) +
  `FakeBrowserExecutor` (записывает вызовы), в стиле `tests/test_agent_graph.py`.
- `tests/test_browser.py`: `test_submit_application` удаляется; тесты whitelist
  обновляются под `allowed_domains`; добавляются тесты `scroll`/`back`.
- `tests/test_agent_graph.py` не меняется (searcher инжектится как фейк).

## Вне области (YAGNI)

- Переписка с работодателем (`send_message`) — не трогаем.
- Пагинация/скроллинг всех страниц результатов — MVP ограничивается тем, что
  собирает цикл за `BROWSER_MAX_STEPS`.
- Общие (не per-user) учётные данные в `JobSite.credentials` — не используем;
  вход под аккаунтом обеспечивается режимом CDP.
