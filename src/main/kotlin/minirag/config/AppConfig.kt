package minirag.config

const val EMBED_MODEL = "bge-m3"
const val CHAT_MODEL = "gemma4:latest"
const val CHUNK_SIZE = 2500
const val OVERLAP = 300
const val MIN_RERANK_WINDOW_SIZE = 500
const val MIN_RERANK_OVERLAP = 100
/*
 * Per-subquery breadth для candidate pool (см. buildCandidatePool).
 * Составной вопрос бьётся на несколько подзапросов, и КАЖДЫЙ
 * подзапрос ищет свой top-N независимо — поэтому top-N должен
 * быть достаточно широким, чтобы найти нужный chunk среди
 * похожих по структуре разделов других сущностей документа
 * (например, "Симптомы" одной болезни среди "Симптомы" десятка
 * других болезней в том же справочнике). Слишком узкий top-N
 * здесь — это баг recall'а: нужный chunk может вообще не попасть
 * в candidate pool, и тогда reranker (который отлично умеет его
 * отличить от нерелевантных) просто никогда его не увидит.
 */
const val TOP_RETRIEVAL = 8
const val TOP_LEXICAL = 8

/*
 * Составные вопросы ("какие симптомы И меры борьбы") бьются
 * Query Analysis на подзапросы (см. minirag.retrieval.buildCandidatePool),
 * и каждый подзапрос вносит своих кандидатов в общий pool.
 * Поэтому финальный отбор после RelevanceGate ограничивается не
 * плоским top-N (это систематически вырезало один из разделов,
 * если его reranker score был чуть ниже другого), а лимитом НА
 * РАЗДЕЛ — так каждый релевантный раздел документа гарантированно
 * долетает до контекста, если он вообще прошёл gate.
 *
 * Лимит считается на пару (title, section), а не на голое
 * section — справочник переиспользует одни и те же названия
 * разделов ("Симптомы", "Меры борьбы") для КАЖДОЙ болезни/сущности,
 * поэтому капать по одному лишь section значило бы, что "Симптомы"
 * одной болезни вытесняют слот у "Симптомы" другой болезни в общем
 * составном ответе (см. AgroKnowledgeTool.capPerSection).
 */
const val MAX_CHUNKS_PER_SECTION = 2
const val MAX_CONTEXT_CHUNKS = 6

/*
 * BGE reranker принимает максимум 512 токенов
 * на пару query + passage.
 */
const val RERANK_MAX_TOKENS = 512

/*
 * Порог принятия chunk как релевантного, в пространстве
 * sigmoid(raw reranker score) — т.е. вероятностной, а не
 * "сырой" logit-шкале, поэтому одно и то же число сравнимо
 * между разными запросами.
 *
 * 0.5f здесь — это НЕ магическое число из головы, а нейтральная
 * точка sigmoid до калибровки (raw score == 0). Реальное значение
 * для этой модели/корпуса должно быть получено через
 * minirag.eval.RerankerCalibration, которая прогоняет reranker
 * на eval-запросах с известным ожидаемым relevant/not-relevant
 * результатом (см. minirag.eval.evalCases и minirag.eval.negativeEvalQueries)
 * и подбирает порог по фактическому разрыву между их score.
 * Замените это значение на то, что выведет калибровка.
 */
const val RELEVANCE_THRESHOLD = 0.3396f

const val OLLAMA_URL = "http://localhost:11434"

/*
 * Без явного num_ctx Ollama сама решает размер контекстного окна
 * (по умолчанию/настройкам модели) — на этой машине это оказалось
 * 65536, из-за чего KV-cache не влезал в 6GB VRAM видеокарты и
 * ~72% слоёв модели уходило на CPU (замерено через `ollama ps`:
 * "72%/28% CPU/GPU"). С явным num_ctx=8192 — "100% GPU", и то же
 * самое обращение к модели идёт заметно быстрее. 8192 — с запасом
 * относительно system prompt + до 6 chunks retrieved context +
 * эхо tool result в истории диалога агента, но всё ещё помещается
 * в VRAM целиком.
 */
const val OLLAMA_NUM_CTX = 8192L

/*
 * ЗАДЕПРЕЧЕНО: аккаунт Yandex Cloud требует верификацию оплаты, которая
 * отклоняет украинские номера телефона и виртуальные/сервисные номера
 * ("введите реальные данные") — пополнить баланс из Украины не удаётся.
 * Оставлено только как справка/на случай если ограничение снимут;
 * см. minirag.web.YandexSearchClient. Активный провайдер — Serper
 * (см. SERPER_* ниже).
 */
const val YANDEX_SEARCH_URL =
    "https://searchapi.api.cloud.yandex.net/v2/web/search"

const val YANDEX_SEARCH_API_KEY_ENV =
    "YANDEX_SEARCH_API_KEY"

const val YANDEX_SEARCH_FOLDER_ID_ENV =
    "YANDEX_SEARCH_FOLDER_ID"

const val YANDEX_SEARCH_REGION_ENV =
    "YANDEX_SEARCH_REGION"

const val YANDEX_SEARCH_TYPE_ENV =
    "YANDEX_SEARCH_TYPE"

/*
 * /shopping, а не /search: для сценариев этого тула (цена/наличие/
 * где купить) обычный органический поиск для целевого региона был
 * либо пуст (gl=by), либо в основном мусор — блоги/соцсети/нерелевант
 * (gl=ru с полным вопросом). Google Shopping возвращает структурированные
 * карточки товаров (цена+магазин) и естественным образом не содержит
 * соцсетей — на живом тесте с gl=ua дал 40/40 релевантных результатов.
 */
const val SERPER_SHOPPING_URL =
    "https://google.serper.dev/shopping"

const val SERPER_API_KEY_ENV =
    "SERPER_API_KEY"

/*
 * gl/hl — страна и язык выдачи Google (см. документацию Serper).
 * "by"/"ru" по умолчанию: подтверждено рабочим запросом из Беларуси,
 * справочник и целевая аудитория — на русском.
 */
const val SERPER_GL_ENV = "SERPER_GL"
const val SERPER_HL_ENV = "SERPER_HL"
/*
 * "by" давал почти пустую/нерелевантную выдачу даже для точечных
 * товарных запросов ("фунгицид фитофтора купить" -> 2 нерелевантных
 * результата, один с украинского домена) — похоже, Google слабо
 * индексирует белорусские магазины. "ua" ближе к реальному
 * местоположению пользователя.
 */
const val SERPER_DEFAULT_GL = "ua"
const val SERPER_DEFAULT_HL = "ru"

const val SERPER_DEFAULT_RECENCY = "w"

const val WEB_SEARCH_DEFAULT_LIMIT = 5
