package minirag.agents

import ai.koog.agents.core.tools.Tool
import ai.koog.agents.core.tools.annotations.LLMDescription
import ai.koog.serialization.JSONSerializer
import ai.koog.serialization.typeToken
import minirag.config.WEB_SEARCH_DEFAULT_LIMIT
import minirag.models.WebSearchArgs
import minirag.web.WebSearchClient

private val NO_RESULTS_MESSAGE = """
    Поиск не вернул результатов.
    Не делай выводов о цене или наличии.
""".trimIndent()

@LLMDescription(
    "Ищет актуальные товарные предложения (цены, магазины) в интернете."
)
class WebSearchTool(
    private val searchClient: WebSearchClient
) : Tool<WebSearchArgs, String>(
    argsType = typeToken<WebSearchArgs>(),
    resultType = typeToken<String>(),
    name = "webSearch",
    description = """
        Ищет актуальные товарные предложения в интернете (Google Shopping).

        Используй webSearch для:
        - цены препаратов;
        - наличия препаратов;
        - где купить препарат;
        - актуальных предложений продавцов;
        - текущей доступности средств защиты растений.

        Не используй webSearch для ответа на вопрос
        о содержании локального агрономического документа.
        Для этого используй searchKnowledge.

        Если вопрос требует одновременно информации
        из локального документа и актуальной информации
        из интернета, используй оба инструмента.

        query должен быть КОРОТКИМ товарным запросом (2-5 слов:
        тип средства/действующее вещество + от чего/для чего + "купить"),
        а НЕ полным вопросом пользователя. Например, вопрос
        "Какие препараты для борьбы с фитофторозом сейчас доступны?"
        нужно превратить в query "фунгицид фитофтороз купить" —
        поиск по товарным каталогам плохо работает с вопросом-
        предложением и вместо магазинов начинает возвращать блоги.

        Если болезнь ВИРУСНАЯ (нет прямого химического препарата,
        который лечит сам вирус), НЕ формируй query по названию
        вируса/болезни — такой запрос ("препараты от вирусной
        болезни купить") даёт нерелевантный мусор, так как под него
        нет товарной категории. Вместо этого возьми из результата
        searchKnowledge переносчика возбудителя (насекомое-
        переносчик, например белокрылка, тля, трипс) и сформируй
        query на средство против НЕГО: тип средства (инсектицид/
        акарицид) + переносчик + "купить". Например, для крапчатости
        листьев томата (возбудитель — вирус, переносчик — белокрылка)
        query должен быть "инсектицид белокрылка купить", а не
        "препараты вирус томата купить". Если переносчик неизвестен
        или отсутствует, честно скажи в ответе, что подходящего
        препарата на рынке нет, вместо того чтобы искать наугад.

        Параметр recency задаёт свежесть результатов:
        - "w" — за последнюю неделю (по умолчанию, большинство вопросов);
        - "d" — за последние сутки, ТОЛЬКО для вопросов о реальном
          событии/новости ("что произошло сегодня");
        - "m" — за последний месяц, если и "w" дал мало результатов.

        Цену/наличие из результата НЕ выдавай как гарантированный
        факт — это данные каталога Google Shopping на момент поиска,
        реальная цена в магазине может отличаться.

        Для КАЖДОГО препарата в ответе указывай: название, цену
        с оговоркой ("ориентировочно"), магазин и ОБЯЗАТЕЛЬНО прямую
        ссылку (поле URL из результата) в формате Markdown, например:
        "[Название препарата](URL) — ориентировочно 45 грн, магазин
        GardenTime". Без ссылки пользователь не сможет проверить
        и купить предложение, поэтому пропускать её нельзя.

        В тексте ссылки (то, что в квадратных скобках) указывай
        КОРОТКОЕ название препарата (поле Title результата, без
        лишних подробностей из него), а НЕ саму ссылку URL — то есть
        "[Карбіон](URL)", а НЕ "[URL](URL)". Длинный URL как текст
        ссылки нечитаем для пользователя.
    """.trimIndent()
) {

    var lastResult: String? = null
        private set

    fun clearLastResult() {
        lastResult = null
    }

    override suspend fun execute(
        args: WebSearchArgs
    ): String {

        println()
        println("========== WEB SEARCH TOOL ==========")
        println("query: ${args.query}")
        println("=====================================")

        /*
         * Внешний API (Serper) — не наш код, может упасть
         * из-за сети, лимитов, просроченной оплаты и т.п. Это
         * граница системы, поэтому ошибка ловится здесь и
         * превращается в честный результат для LLM, а не роняет
         * весь прогон агента.
         */
        val results =
            try {
                searchClient.search(
                    query = args.query,
                    limit = WEB_SEARCH_DEFAULT_LIMIT,
                    recency = args.recency
                )
            } catch (e: Exception) {
                println("[webSearch] search failed: ${e.message}")
                val result = """
                    Веб-поиск сейчас недоступен (${e.message}).
                    Не делай выводов о цене или наличии из своих знаний —
                    сообщи пользователю, что актуальные данные получить
                    не удалось.
                """.trimIndent()
                lastResult = result
                return result
            }

        if (results.isEmpty()) {
            lastResult = NO_RESULTS_MESSAGE
            return NO_RESULTS_MESSAGE
        }

        val result = buildString {
            append("WEB SEARCH RESULTS\n\n")
            results.forEachIndexed { index, result ->
                append("[${index + 1}]\n")
                append("Title: ${result.title}\n")
                append("URL: ${result.url}\n")

                if (result.snippet.isNotBlank()) {
                    append("Snippet: ${result.snippet}")
                    append("\n")
                }
                append("\n")
            }
        }
        lastResult = result
        return result
    }

    override fun encodeResultToString(
        result: String,
        serializer: JSONSerializer
    ): String {
        return result
    }
}
