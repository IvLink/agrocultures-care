package minirag.models

data class DocumentChunk(
    val id: Int,
    val page: Int,
    val text: String,
    val section: String? = null,
    /*
     * Заголовок записи (например, название болезни), к которой
     * относится chunk. Заполняется chunker'ом, когда title
     * страницы физически "уехал" не туда из-за порядка извлечения
     * текста из PDF (см. TextChunker.splitIntoSectionBlocks) —
     * без этого якоря retrieval/reranker не видят, что чанк
     * относится к другой сущности, чем соседние по разделу.
     */
    val title: String? = null
) {

    /*
     * Текст, который реально должен видеть эмбеддинг/reranker:
     * с title в качестве явного якоря в начале, а не зарытым
     * где-то в середине текста.
     */
    fun retrievalText(): String =
        title?.let { "$it\n\n$text" } ?: text
}