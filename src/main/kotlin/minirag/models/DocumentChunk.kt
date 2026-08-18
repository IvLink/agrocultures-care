package minirag.models

data class DocumentChunk(
    val id: Int,
    val page: Int,
    val text: String,
    val section: String? = null
)