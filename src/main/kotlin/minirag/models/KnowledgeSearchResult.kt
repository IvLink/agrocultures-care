package minirag.models

data class KnowledgeSearchResult(
    val query: String,
    val chunks: List<String>
)