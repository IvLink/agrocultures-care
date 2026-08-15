package minirag.agents

import ai.koog.agents.core.tools.annotations.LLMDescription
import ai.koog.agents.core.tools.annotations.Tool
import ai.koog.agents.core.tools.reflect.ToolSet
import minirag.bge_reranker.BgeReranker
import minirag.config.TOP_RERANK
import minirag.retrieval.Retriever

@LLMDescription(
    "Инструменты для поиска информации в локальной базе знаний " +
            "по болезням растений, симптомам, диагностике и лечению."
)
class AgroKnowledgeTool(
    private val retriever: Retriever,
    private val reranker: BgeReranker,
    private val chunks: List<String>,
    private val chunkVecs: List<List<Double>>
) : ToolSet {

    @Tool
    @LLMDescription(
        "Ищет в локальной базе знаний информацию о симптомах, " +
                "диагностике и лечении болезней растений. " +
                "Используй этот инструмент, когда для ответа нужна " +
                "информация из агро-справочников."
    )
    suspend fun searchKnowledge(
        @LLMDescription(
            "Конкретный вопрос для поиска в базе знаний. " +
                    "Например: 'Какие симптомы фитофтороза томата?' " +
                    "или 'Чем лечить фитофтороз томата?'"
        )
        query: String
    ): String {

        println("\n[tool] searchKnowledge")
        println("[tool] query: $query")

        val candidates = retriever
            .multiQueryRetrieval(
                questions = listOf(query),
                chunks = chunks,
                chunkVecs = chunkVecs
            )
            .distinct()

        println("[tool] candidates: ${candidates.size}")

        val reranked = reranker.multiQueryRerank(
            queries = listOf(query),
            chunks = candidates,
            topN = TOP_RERANK
        )
        println("\n========== TOOL RESULT ==========")
        val toolResult = reranked
            .flatMap { it.chunks }
            .joinToString("\n\n---\n\n")
            .ifBlank {
                "По этому запросу в базе знаний ничего не найдено."
            }

        println(toolResult)
        println("========== END TOOL RESULT ==========\n")
        return toolResult
    }
}