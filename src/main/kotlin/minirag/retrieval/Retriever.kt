package minirag.retrieval

import minirag.config.TOP_RETRIEVAL
import minirag.extensions.cosineSimilarity
import minirag.ollama.OllamaClient

class Retriever(private val ollama: OllamaClient) {

    suspend fun topChunks(
        question: String,
        chunks: List<String>,
        chunkVecs: List<List<Double>>
    ): List<String> {
        val qVec = ollama.embed(listOf(question))[0]
        val similarities = chunkVecs.map { it.cosineSimilarity(qVec) }
        val bestIds = similarities
            .indices
            .sortedByDescending { similarities[it] }
            .take(TOP_RETRIEVAL)

        bestIds.forEachIndexed { index, id ->
            println(
                """
            ===== CHUNK $index / id=$id =====
            similarity = ${"%.3f".format(similarities[id])}

            ${chunks[id]}

            """.trimIndent()
            )
        }

        return bestIds.map { chunks[it] }
    }

    suspend fun multiQueryRetrieval(
        questions: List<String>,
        chunks: List<String>,
        chunkVecs: List<List<Double>>
    ): List<String> {
        val allResults = mutableListOf<String>()
        for (query in questions) {
            println("\n[query] $query")
            val results = topChunks(
                question = query,
                chunks = chunks,
                chunkVecs = chunkVecs
            )
            allResults += results
        }
        return allResults.distinct()
    }
}
