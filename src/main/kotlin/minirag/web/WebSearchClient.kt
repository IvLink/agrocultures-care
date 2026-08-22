package minirag.web

import minirag.web.models.WebSearchResult

interface WebSearchClient {

    /*
     * recency: "d" (сутки) / "w" (неделя) / "m" (месяц) — фильтр
     * свежести результатов (Google "qdr", см. SerperSearchClient).
     * null/неизвестное значение -> реализация сама выбирает дефолт.
     */
    suspend fun search(
        query: String,
        limit: Int = 5,
        recency: String? = null
    ): List<WebSearchResult>
}