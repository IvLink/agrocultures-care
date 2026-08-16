package minirag.models

data class RerankedChunk(
    val chunkId: Int,
    val score: Float
)

data class RerankResult(
    val query: String,
    val chunks: List<RerankedChunk>
)