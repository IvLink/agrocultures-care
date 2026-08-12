package minirag.models

data class RerankResult(
    val query: String,
    val chunks: List<String>
)