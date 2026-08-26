# Реестр способностей, загрузка резюме и RAG-скоринг

Дата: 2026-08-26

## Цель

Сейчас агент ищет и скорит вакансии «вслепую»: таблица `profiles` пуста, пути
заполнить её нет, а RAG-код (`src/rag/ingest.py`, `src/rag/retrieve.py`) написан,
но нигде не вызывается. Дополнительно роутинг намерений жёстко зашит в графе
(`Literal["search_job", "stats", ...]` + `route_by_intent`), поэтому добавить
новую возможность можно только правкой enum + узла + рёбер.

Цель — три связанных изменения:

1. **Реестр способностей** — агент получает список того, что умеет (имя +
   описание), и по сообщению либо вызывает нужный обработчик, либо честно
   отвечает, что такого функционала нет.
2. **Загрузка резюме** — пользователь присылает PDF/DOCX в Telegram, агент
   извлекает текст и структурирует профиль.
3. **RAG-скоринг** — резюме чанкуется и эмбеддится; при оценке вакансии
   извлекаются релевантные фрагменты резюме и подмешиваются в промпт скоринга.

Механизм выбора способности — вариант B: реестр + `invoke_structured` (тот же
кирпич, что уже используется в `IntentRouter`), а не нативный `bind_tools`, чтобы
работало на Ollama (`llama3.2`), где tool-calling нестабилен.

## Требования

1. Пользователь может спросить «куда прислать резюме» — агент отвечает
   «пришли PDF/DOCX сюда» и ждёт файл.
2. Присланный документ парсится: текст извлекается, структура профиля
   (`ProfileData`) сохраняется в `profiles`, чанки с эмбеддингами — в `documents`.
3. При оценке вакансии в промпт скоринга добавляются релевантные фрагменты
   резюме (RAG), наряду со структурированным профилем.
4. Запрос, которому не соответствует ни одна способность (например, «отправь мне
   письмо на почту»), отвечает «такого функционала пока нет» с перечнем умений.
5. Всё работает на `ollama` (дефолт `.env`) и не ломается при недоступности LLM
   (детерминированный keyword-fallback).
6. Размерность эмбеддингов согласована: ollama `nomic-embed-text` = 768.

## Архитектура

Два слоя вместо одного графа-роутера:

- **Диспетчер способностей** — верхний слой, выбирает capability по тексту.
- **Граф поиска** — суженный `build_graph`, только пайплайн
  `search → match → decision → apply → report` + HITL.

Новые/изменённые компоненты:

- `src/agent/capabilities.py` (новый) — `Capability`, `CapabilityChoice`,
  `CapabilityRouter`, сборка реестра.
- `src/agent/graph.py` — из графа убираются `router_node`, `route_by_intent`,
  `stats_node`, `chat_node`; добавляется `retriever` в `match_node`.
- `src/rag/profile_repo.py` (новый) — `ProfileRepository` (upsert `profiles`).
- `src/rag/retrieve.py` — добавляется обёртка `retrieve_for_scoring` /
  класс-ретривер (эмбеддинг запроса + `retrieve_relevant`).
- `src/llm/gateway.py` — `cached_property embedder` через `build_embeddings`.
- `src/tg/service.py` — диспетчеризация по capability; `handle_document`.
- `src/tg/bot.py` — хендлер документов (`F.content_type == "document"`),
  композиционный корень (реестр, ретривер, репозиторий профиля).
- `migrations/versions/<hash>_align_embedding_dim.py` (новый, hash генерирует
  alembic) — `ALTER COLUMN embedding TYPE vector(768)`.

## Реестр способностей

```python
class Capability(BaseModel):
    key: str                 # уникальный ключ
    description: str         # описание для LLM

class CapabilityChoice(BaseModel):
    key: str | None          # None — ни одна не подходит
```

Обработчики унифицированы сигнатурой `async (user_id: int, message: str) -> str`
(возвращают текст ответа пользователю). Сопоставление `key → handler` собирается
в композиционном корне; `Capability` хранит только метаданные для LLM-промпта.

`CapabilityRouter` аналогичен `IntentRouter`:

```python
async def classify(self, message: str) -> str | None:
    # invoke_structured с промптом, перечисляющим ключи+описания
    # при Exception — keyword-fallback по ключевым словам
```

Промпт:

> Ты — маршрутизатор личного агента поиска работы. Доступные возможности:
> - upload_resume — принять и разобрать резюме (PDF/DOCX)
> - search_job — искать/откликаться на вакансии
> - stats — показать статистику откликов
> Верни ключ одной из возможностей или "none", если ни одна не подходит.

Реестр собирается в `build_tg_service` (композиционный корень): ключ →
обработчик. Ключи:

- `search_job` → граф поиска (инъекция `searcher`/`applier`/`profile_provider`).
- `stats` → `get_user_stats`.
- `upload_resume` → текстовый ответ «пришли PDF/DOCX сюда».
- `None` → «такого функционала пока нет. Я умею: …» (перечень из реестра).

## Диспетчер в `TgService.handle_message`

Порядок обработки входящего сообщения:

1. **Документ** (`handle_document`, вызывается из хендлера `bot.py`) — парсинг
   резюме, без LLM-роутинга.
2. **HITL pending** — если граф приостановлен (`aget_state` + `snapshot.next`),
   возобновить `Command(resume=...)` как сейчас.
3. **Onboarding** — нет сайтов → спросить/сохранить сайты (без изменений).
4. **Текст** → `CapabilityRouter.classify` → key:
   - `search_job` → `graph.ainvoke` (состояние без поля `intent`);
   - `stats` → обработчик статистики;
   - `upload_resume` → «пришли PDF/DOCX сюда»;
   - `None` → «такого функционала пока нет».

## Загрузка резюме

`bot.py` добавляет хендлер `@dp.message(F.content_type == "document")`:
скачивает файл через `bot.download`/`bot.get_file`, вызывает
`service.handle_document(user_id, filename, content: bytes)`.

`handle_document`:

1. Проверить расширение (`.pdf`/`.docx`), иначе «пришли PDF или DOCX».
2. Записать `content` во временный файл, вызвать `extract_cv_text(path)`.
3. Пустой текст → «не удалось прочитать документ».
4. `extract_profile(gateway, cv_text)` → `ProfileData`.
5. `ProfileRepository.save(user_id, profile_data)` (upsert по `user_id`).
6. `chunk_by_sections(cv_text)` → `embed_chunks(embedder, chunks)` →
   `store_chunks(session, user_id, chunks, embeddings)`.
7. Ответ: «Записал профиль: роль …, навыки …, опыт …».

`ProfileRepository` (в `src/rag/profile_repo.py`):

```python
class ProfileRepository:
    async def save(self, user_id: int, data: ProfileData) -> None:
        # upsert: найти Profile по user_id, обновить или создать
```

## RAG-скоринг

`build_graph` получает новую зависимость
`retriever: Callable[[int, str], Awaitable[list[str]]]`.

В `match_node` для каждой вакансии:

1. Запрос = `f"{title} {description}"`.
2. `chunks = await retriever(user_id, query)` (релевантные фрагменты резюме).
3. `_score_candidate(gateway, candidate, profile, chunks)` — промпт дополняется
   блоками:

```
Профиль: {profile}
Релевантные фрагменты резюме:
- {chunk1}
- {chunk2}
Вакансия: {candidate}
```

Ретривер (обёртка в `src/rag/retrieve.py`): эмбеддинг запроса через `embedder`
→ `retrieve_relevant(session, user_id, query_embedding, top_k=5)` → тексты чанков.

`embedder` — `cached_property` на `LLMGateway`, строится через существующий
`build_embeddings(settings.LLM_PROVIDER, settings.EMBEDDING_MODEL,
settings.OPENAI_API_KEY, settings.OLLAMA_BASE_URL)`. Провайдер эмбеддингов
совпадает с `LLM_PROVIDER`.

## Модель данных

Таблицы не меняются. Только миграция размерности:

- `documents.embedding` сейчас `VECTOR(1536)` (захардкожено в
  `ea5108fb8c54_init_models.py`), а `EMBEDDING_DIM=768` (ollama `nomic-embed-text`).
- Новая миграция: `ALTER TABLE documents ALTER COLUMN embedding TYPE vector(768)`.
- ORM уже читает `Vector(settings.EMBEDDING_DIM)` — после миграции согласуется.

## Обработка ошибок

- `CapabilityRouter`: при ошибке LLM — keyword-fallback (как в `IntentRouter`).
- Неподдерживаемый формат документа → «пришли PDF или DOCX».
- Пустой текст / сбой парсинга → «не удалось прочитать документ».
- Эмбеддинги недоступны (Ollama не запущена / модель не скачана) → загрузка
  профиля продолжается (структура сохраняется), чанки не пишутся; скоринг
  работает без RAG-фрагментов (пустой список).

## Безопасность

- HITL-подтверждение для `apply` сохраняется.
- Резюме обрабатывается только в per-user скоупе (`user_id`), без утечки между
  пользователями (`retrieve_relevant` и `store_chunks` фильтруют по `user_id`).
- Гардрейлы браузера и исходящих сообщений не затрагиваются.

## Тестирование

- `CapabilityRouter`: выбор ключа по фейку, keyword-fallback, `None`.
- Диспетчер: «отправь на почту» → «не умею»; «куда прислать резюме» → «пришли».
- Хендлер резюме: fake-`extract_cv_text`/`extract_profile` → `ProfileRepository`
  и `store_chunks` вызываются с ожидаемыми данными; неверный формат → отказ.
- RAG-скоринг: `retriever` вызывается для каждой вакансии, фрагменты попадают в
  промпт `_score_candidate` (аналог `test_match_uses_profile`).
- `test_agent_graph.py`: убрать `test_chat_flow`; остальные перевести с фейкового
  `intent` на прямой запуск search-пайплайна (граф больше не читает `intent`).
- `test_tg.py`: обновить `FakeGateway` (без `Intent`), добавить кейсы диспетчера.

## Вне области (YAGNI)

- Переписка с работодателем (`send_message`) — не трогаем.
- Нативный `bind_tools`/`ToolNode` — не используем (Ollama).
- Многооборотное состояние «жду файл» — не нужно: документ однозначен по типу
  сообщения, отдельный роутинг не требуется.
- Отдельный `EMBEDDING_PROVIDER` — провайдер эмбеддингов = `LLM_PROVIDER`.
- Пересоздание эмбеддингов при смене `EMBEDDING_DIM` — ручная операция (README),
  не автоматизируем.
