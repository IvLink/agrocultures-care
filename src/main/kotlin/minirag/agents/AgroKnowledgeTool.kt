package minirag.agents

import ai.koog.agents.core.tools.ToolCallMetadata
import ai.koog.agents.core.tools.annotations.LLMDescription
import ai.koog.serialization.JSONSerializer
import ai.koog.serialization.typeToken
import minirag.bge_reranker.BgeReranker
import minirag.config.TOP_RERANK
import minirag.models.AgroSearchArgs
import minirag.retrieval.Retriever
import ai.koog.agents.core.tools.Tool

@LLMDescription(
    "Инструменты для поиска информации в локальной базе знаний " +
            "по болезням растений, симптомам, диагностике и лечению."
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
        Ищет в локальной базе знаний информацию о симптомах,
        диагностике и лечении болезней растений.

        Используй этот инструмент, когда для ответа нужна
        информация из агро-справочников.
    """.trimIndent()
) {

    var lastResult: String? = null
        private set

    fun clearLastResult() {
        lastResult = null
    }

    override suspend fun execute(args: AgroSearchArgs): String {

        val query = args.query

        println()
        println("========== SEARCH KNOWLEDGE ==========")
        println("query: $query")
        println("======================================")

        val candidates = retriever
            .multiQueryRetrieval(
                questions = listOf(query),
                chunks = chunks,
                chunkVecs = chunkVecs
            )
            .distinct()

        println("[searchKnowledge] candidates: ${candidates.size}")

        val reranked = reranker.multiQueryRerank(
            queries = listOf(query),
            chunks = candidates,
            topN = TOP_RERANK
        )

        val result = reranked
            .flatMap { it.chunks }
            .distinct()
            .joinToString("\n\n---\n\n")
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
