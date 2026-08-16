package minirag.retrieval

import minirag.config.TOP_RETRIEVAL
import minirag.extensions.cosineSimilarity
import minirag.models.RetrievedChunk
import minirag.ollama.OllamaClient

class Retriever(private val ollama: OllamaClient) {
    suspend fun topChunks(
        question: String,
        chunks: List<String>,
        chunkVecs: List<List<Double>>
    ): List<RetrievedChunk> {

        require(chunks.size == chunkVecs.size) {
            "chunks.size=${chunks.size} but chunkVecs.size=${chunkVecs.size}"
        }

        if (chunks.isEmpty()) {
            return emptyList()
        }

        val qVec = ollama.embed(listOf(question))[0]

        val similarities = chunkVecs.map { vector ->
            vector.cosineSimilarity(qVec)
        }

        return similarities
            .indices
            .sortedByDescending { similarities[it] }
            .take(TOP_RETRIEVAL)
            .map { chunkId ->
                RetrievedChunk(
                    chunkId = chunkId,
                    similarity = similarities[chunkId]
                )
            }
    }

    suspend fun multiQueryRetrieval(
        questions: List<String>,
        chunks: List<String>,
        chunkVecs: List<List<Double>>
    ): List<RetrievedChunk> {

        val allResults = mutableListOf<RetrievedChunk>()

        for (query in questions) {

            val results = topChunks(
                question = query,
                chunks = chunks,
                chunkVecs = chunkVecs
            )

            allResults += results
        }

        /*
         * Один и тот же chunk может попасть в результаты
         * нескольких query. Оставляем его один раз.
         *
         * При этом сохраняем его лучший similarity.
         */
        return allResults
            .groupBy { it.chunkId }
            .values
            .map { matches ->
                matches.maxBy { it.similarity }
            }
            .sortedByDescending { it.similarity }
    }
}
