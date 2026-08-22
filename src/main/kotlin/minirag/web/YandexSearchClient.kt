package minirag.web

import kotlinx.serialization.json.Json
import minirag.web.models.WebSearchResult
import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.request.header
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.contentType
import minirag.config.YANDEX_SEARCH_URL
import minirag.web.models.YandexGroupSpec
import minirag.web.models.YandexSearchQuery
import minirag.web.models.YandexSearchRequest
import minirag.web.models.YandexSearchResponse
import org.jsoup.Jsoup
import kotlin.io.encoding.Base64

/*
 * ЗАДЕПРЕЧЕНО: Yandex Cloud требует верификацию оплаты для активации
 * Search API, а верификация отклоняет украинские номера телефона и
 * виртуальные/сервисные номера ("введите реальные данные") — то есть
 * из Украины пополнить баланс и получить рабочий ключ нельзя. Кроме
 * того, реальная схема FORMAT_XML-ответа так и не была подтверждена
 * живым запросом (см. parseResults ниже — написан вслепую под старый
 * FORMAT_HTML и почти наверняка не разбирает XML). Класс оставлен в
 * коде как справка/на случай снятия ограничения, но активный провайдер
 * — minirag.web.SerperSearchClient (Serper.dev, подтверждён рабочим
 * из Беларуси, не требует карты для free tier).
 */
@Deprecated(
    message = "Yandex Search API недоступен из Украины (верификация оплаты " +
            "отклоняет украинские номера) и XML-парсер не проверен на реальных " +
            "данных. Используйте minirag.web.SerperSearchClient.",
    replaceWith = ReplaceWith("SerperSearchClient")
)
class YandexSearchClient(
    private val httpClient: HttpClient,
    private val apiKey: String,
    private val folderId: String,
    private val searchType: String,
    private val region: String?
) : WebSearchClient {

    private val json = Json { ignoreUnknownKeys = true }

    override suspend fun search(
        query: String,
        limit: Int,
        recency: String?
    ): List<WebSearchResult> {

        require(query.isNotBlank()) { "Search query must not be blank" }
        /*
         * groupsOnPage: "от 1 до 100 для XML, от 5 до 50 для HTML"
         * (документация Yandex Search API). Мы запрашиваем FORMAT_XML,
         * поэтому диапазон 1..100.
         */
        require(limit in 1..100) { "Search limit must be between 1 and 100 for FORMAT_XML" }
        println()
        println("========== YANDEX SEARCH ==========")
        println("query: $query")
        println("searchType: $searchType")
        println("region: ${region ?: "default"}")
        println("===================================")

        val request = YandexSearchRequest(
            query = YandexSearchQuery(
                searchType = searchType,
                queryText = query
            ),
            groupSpec = YandexGroupSpec(
                groupsOnPage = limit.toString()
            ),
            region = region,
            folderId = folderId,
            /*
             * FORMAT_XML, а не FORMAT_HTML: по документации XML содержит
             * только результаты поиска без рекламы/быстрых ответов/прочего
             * SERP-мусора. Реальную схему XML-ответа мы пока не видели
             * (страница response.html редиректит на web-search.html —
             * отдельной схемы в доке нет), поэтому parseResults() ниже
             * пока не умеет её разбирать и просто вернёт пустой список.
             * Как только появится реальный запрос — сырое decoded-тело
             * пишется в консоль (см. ниже), и по нему уже пишем парсер.
             */
            responseFormat = "FORMAT_XML"
        )

        val response =
            httpClient.post(YANDEX_SEARCH_URL) {
                header(
                    HttpHeaders.Authorization,
                    "Api-Key $apiKey"
                )
                contentType(ContentType.Application.Json)
                setBody(request)
            }
                .body<YandexSearchResponse>()

        val rawData = response.rawData ?: return emptyList()

        val rawBody =
            try {
                Base64
                    .decode(rawData)
                    .toString(Charsets.UTF_8)
            } catch (e: IllegalArgumentException) {
                error(
                    "Yandex Search returned invalid Base64"
                )
            }

        /*
         * Диагностический дамп: реальной схемы XML-ответа в доке нет
         * (response.html редиректит на web-search.html), поэтому пока
         * единственный способ её увидеть — прогнать живой запрос и
         * посмотреть сюда. Убрать/урезать, когда парсер под XML будет
         * написан по факту.
         */
        println()
        println("========== YANDEX RAW RESPONSE (decoded) ==========")
        println(rawBody)
        println("====================================================")

        return parseResults(
            body = rawBody,
            limit = limit
        )
    }

    /*
     * TODO: написан под старый FORMAT_HTML (поиск <a href>) и с
     * FORMAT_XML почти наверняка будет возвращать пустой список —
     * реальных <a>-тегов в XML-ответе не ожидается. Оставлен как
     * заглушка, которая не падает, пока не появится реальный XML
     * (см. дамп в search()) и по нему не будет переписан разбор
     * под настоящие теги ответа.
     */
    private fun parseResults(
        body: String,
        limit: Int
    ): List<WebSearchResult> {

        val document = Jsoup.parse(body)

        val results = document
            .select("a")
            .mapNotNull { element ->

                val href = element
                    .attr("href")
                    .trim()
                val title = element
                    .text()
                    .trim()

                if (href.isBlank() || title.isBlank()) {
                    return@mapNotNull null
                }

                if (
                    !href.startsWith("http://") &&
                    !href.startsWith("https://")
                ) {
                    return@mapNotNull null
                }

                WebSearchResult(
                    title = title,
                    url = href,
                    snippet = ""
                )
            }
            .distinctBy { it.url }
            .take(limit)

        /*
         * Если структура ответа изменилась,
         * лучше вернуть пустой результат, чем
         * скормить LLM мусор.
         */
        if (results.isEmpty()) {
            println("[webSearch] no parsable results")
            return emptyList()
        }

        println("[webSearch] results: ${results.size}")
        results.forEachIndexed { index, result ->
            println("[${index + 1}] " + result.title)
            println("    ${result.url}")
        }

        return results
    }
}