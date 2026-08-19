package minirag.agents

import ai.koog.agents.core.tools.Tool
import ai.koog.agents.core.tools.annotations.LLMDescription
import ai.koog.serialization.JSONSerializer
import ai.koog.serialization.typeToken
import kotlin.system.measureTimeMillis
import minirag.config.MAX_CHUNKS_PER_SECTION
import minirag.config.MAX_CONTEXT_CHUNKS
import minirag.models.AgroSearchArgs
import minirag.models.DocumentChunk
import minirag.models.RerankResult
import minirag.models.RetrievedChunk
import minirag.ollama.OllamaClient
import minirag.reranker.BgeReranker
import minirag.retrieval.LexicalRetriever
import minirag.retrieval.NO_RELEVANT_CONTEXT_MESSAGE
import minirag.retrieval.Retriever
import minirag.retrieval.buildCandidatePool
import minirag.retrieval.buildContext
import minirag.retrieval.gateRelevant

@LLMDescription(
    "Инструмент поиска информации в локальной базе знаний."
)
class AgroKnowledgeTool(
    private val ollama: OllamaClient,
    private val retriever: Retriever,
    private val lexicalRetriever: LexicalRetriever,
    private val reranker: BgeReranker,
    private val chunks: List<DocumentChunk>,
    private val chunkVecs: List<List<Double>>,
    private val relevanceThreshold: Float
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
         * 1-3. Query analysis -> dense + lexical retrieval
         * по каждому подзапросу -> candidate pool.
         */
        var candidates: List<RetrievedChunk> = emptyList()
        val candidatePoolMs = measureTimeMillis {
            candidates =
                buildCandidatePool(
                    question = query,
                    ollama = ollama,
                    retriever = retriever,
                    lexicalRetriever = lexicalRetriever,
                    chunks = chunks,
                    chunkVecs = chunkVecs
                )
        }

        println(
            "[searchKnowledge] candidate pool: ${candidates.size} " +
                    "(decompose+retrieval: ${candidatePoolMs}мс)"
        )

        if (candidates.isEmpty()) {
            val result = NO_RELEVANT_CONTEXT_MESSAGE
            lastResult = result
            return result
        }

        /*
         * 4. Cross-encoder reranking относительно исходного
         * вопроса + accept/reject по откалиброванному threshold.
         */
        var reranked: List<RerankResult> = emptyList()
        val rerankMs = measureTimeMillis {
            reranked =
                reranker.rerank(
                    query = query,
                    candidates = candidates,
                    chunks = chunks,
                    threshold = relevanceThreshold
                )
        }
        println("[searchKnowledge] reranker: ${rerankMs}мс")

        /*
         * 5. RelevanceGate: RELEVANT / NOT_RELEVANT.
         */
        val relevant = gateRelevant(reranked)
        if (relevant.isEmpty()) {
            val result = NO_RELEVANT_CONTEXT_MESSAGE
            println()
            println("[searchKnowledge] ни один кандидат не прошёл relevance gate")
            lastResult = result
            return result
        }

        /*
         * 6. Отбор для контекста: лимит НА РАЗДЕЛ, а не плоский
         * top-N — иначе один раздел документа систематически
         * вытесняет другой в составных вопросах.
         */
        val chunksById = chunks.associateBy { it.id }
        val selectedResults =
            capPerSection(
                results = relevant,
                chunksById = chunksById,
                maxPerSection = MAX_CHUNKS_PER_SECTION,
                maxTotal = MAX_CONTEXT_CHUNKS
            )
        val selectedChunks = selectedResults.mapNotNull { chunksById[it.chunkId] }

        /*
         * 7. ContextBuilder.
         */
        val result = buildContext(selectedChunks)
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

/*
 * Капается по паре (title, section), а не по голому section:
 * справочник переиспользует одни и те же названия разделов
 * ("Симптомы", "Меры борьбы") для каждой болезни/сущности, поэтому
 * капание по одному section значило бы, что "Симптомы" болезни A
 * и "Симптомы" болезни B делят один и тот же слот — и в составном
 * вопросе один из них может вытеснить другой ещё до того, как
 * реально релевантный chunk другой болезни доберётся до контекста.
 */
@Suppress("SameParameterValue")
private fun capPerSection(
    results: List<RerankResult>,
    chunksById: Map<Int, DocumentChunk>,
    maxPerSection: Int,
    maxTotal: Int
): List<RerankResult> {

    val perSectionCount = mutableMapOf<Pair<String?, String?>, Int>()
    val selected = mutableListOf<RerankResult>()

    for (result in results) {
        if (selected.size >= maxTotal) {
            break
        }
        val chunk = chunksById[result.chunkId]
        val sectionKey = chunk?.title to chunk?.section
        val count = perSectionCount.getOrDefault(sectionKey, 0)
        if (count >= maxPerSection) {
            continue
        }
        selected += result
        perSectionCount[sectionKey] = count + 1
    }

    return selected
}
