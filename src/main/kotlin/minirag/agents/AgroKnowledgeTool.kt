package minirag.agents

import ai.koog.agents.core.tools.Tool
import ai.koog.agents.core.tools.annotations.LLMDescription
import ai.koog.serialization.JSONSerializer
import ai.koog.serialization.typeToken
import minirag.bge_reranker.BgeReranker
import minirag.config.TOP_RERANK
import minirag.models.AgroSearchArgs
import minirag.retrieval.Retriever

@LLMDescription(
    "Инструмент поиска информации в локальной базе знаний."
)
class AgroKnowledgeTool(
    private val retriever: Retriever,
    private val reranker: BgeReranker,
    private val chunks: List<String>,
    private val chunkVecs: List<List<Double>>
) : Tool<AgroSearchArgs, String>(
    argsType = typeToken<AgroSearchArgs>(),
    resultType = typeToken<String>(),
    name = "searchKnowledge",
    description = """
        Ищет информацию в локальной базе знаний.

        Используй этот инструмент, когда для ответа
        нужна информация из загруженных документов.
    """.trimIndent()
) {

    var lastResult: String? = null
        private set

    fun clearLastResult() {
        lastResult = null
    }

    override suspend fun execute(
        args: AgroSearchArgs
    ): String {

        val query = args.query

        println()
        println("========== SEARCH KNOWLEDGE ==========")
        println("query: $query")
        println("======================================")

        /*
         * 1. Vector retrieval.
         *
         * Здесь получаем ID исходных chunks,
         * а не сами строки.
         */
        val candidates = retriever
            .multiQueryRetrieval(
                questions = listOf(query),
                chunks = chunks,
                chunkVecs = chunkVecs
            )

        println(
            "[searchKnowledge] candidates: ${candidates.size}"
        )

        candidates.forEach {
            println(
                "candidate chunk[${it.chunkId}] " +
                        "similarity=${"%.4f".format(it.similarity)}"
            )
        }

        if (candidates.isEmpty()) {

            val result =
                "По этому запросу в базе знаний ничего не найдено."

            lastResult = result

            return result
        }

        /*
         * 2. Reranking.
         *
         * Reranker получает исходные chunkId.
         */
        val reranked = reranker.multiQueryRerank(
            queries = listOf(query),
            candidates = candidates,
            chunks = chunks,
            topN = TOP_RERANK
        )

        /*
         * 3. Собираем финальный context.
         */
        val selectedChunks = reranked
            .flatMap { it.chunks }
            .distinctBy { it.chunkId }

        val result = selectedChunks
            .joinToString("\n\n---\n\n") { selected ->

                buildString {

                    append("[chunkId=${selected.chunkId}]")
                    append("\n")
                    append(chunks[selected.chunkId])
                }
            }
            .ifBlank {
                "По этому запросу в базе знаний ничего не найдено."
            }

        println()
        println("========== END TOOL RESULT ==========")
        println(result)
        println("=====================================")
        println()

        lastResult = result

        return result
    }

    override fun encodeResultToString(
        result: String,
        serializer: JSONSerializer
    ): String {
        return result
    }
}
