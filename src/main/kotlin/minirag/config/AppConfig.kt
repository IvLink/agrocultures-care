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
