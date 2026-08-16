package minirag.models

data class RetrievedChunk(
    val chunkId: Int,
    val similarity: Double
)