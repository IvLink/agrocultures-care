package minirag.config

const val EMBED_MODEL = "bge-m3"
const val CHAT_MODEL = "gemma4:latest"
const val CHUNK_SIZE = 2500
const val OVERLAP = 300
const val MIN_RERANK_WINDOW_SIZE = 500
const val MIN_RERANK_OVERLAP = 100
const val TOP_RETRIEVAL = 5
const val TOP_LEXICAL = 5

/*
 * Составные вопросы ("какие симптомы И меры борьбы") бьются
 * Query Analysis на подзапросы (см. minirag.retrieval.buildCandidatePool),
 * и каждый подзапрос вносит своих кандидатов в общий pool.
 * Поэтому финальный отбор после RelevanceGate ограничивается не
 * плоским top-N (это систематически вырезало один из разделов,
 * если его reranker score был чуть ниже другого), а лимитом НА
 * РАЗДЕЛ — так каждый релевантный раздел документа гарантированно
 * долетает до контекста, если он вообще прошёл gate.
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
const val RELEVANCE_THRESHOLD = 0.5f

const val OLLAMA_URL = "http://localhost:11434"
