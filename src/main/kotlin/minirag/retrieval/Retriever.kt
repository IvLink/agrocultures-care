package minirag.retrieval

import minirag.config.TOP_RETRIEVAL
import minirag.extensions.cosineSimilarity
import minirag.models.DocumentChunk
import minirag.models.RetrievedChunk
import minirag.ollama.OllamaClient

class Retriever(
    private val ollama: OllamaClient
) {

    suspend fun topChunks(
        question: String,
        chunks: List<DocumentChunk>,
        chunkVecs: List<List<Double>>
    ): List<RetrievedChunk> {

        require(chunks.size == chunkVecs.size) {
            "chunks.size=${chunks.size} but chunkVecs.size=${chunkVecs.size}"
        }

        if (chunks.isEmpty()) {
            return emptyList()
        }

        val qVec = ollama.embed(
            listOf(question)
        )[0]

        val similarities = chunkVecs.map { vector ->
            vector.cosineSimilarity(qVec)
        }

        return similarities
            .indices
            .sortedByDescending {
                similarities[it]
            }
            .take(TOP_RETRIEVAL)
            .map { chunkIndex ->

                val chunk = chunks[chunkIndex]

                RetrievedChunk(
                    chunkId = chunk.id,
                    similarity = similarities[chunkIndex]
                )
            }
    }
}