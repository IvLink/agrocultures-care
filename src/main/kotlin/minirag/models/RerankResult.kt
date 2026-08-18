package minirag.models

data class RerankResult(
    val chunkId: Int,
    val rawScore: Float,
    val normalizedScore: Float,
    val accepted: Boolean
)
