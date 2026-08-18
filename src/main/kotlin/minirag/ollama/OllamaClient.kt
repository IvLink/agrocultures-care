package minirag.ollama

import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.plugins.timeout
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.http.ContentType
import io.ktor.http.contentType
import kotlinx.serialization.json.Json
import minirag.config.CHAT_MODEL
import minirag.config.EMBED_MODEL
import minirag.config.OLLAMA_URL
import minirag.extensions.cleanJsonResponse
import minirag.models.ChatMessage
import minirag.models.ChatOptions
import minirag.models.ChatRequest
import minirag.models.ChatResponseMessage
import minirag.models.EmbedRequest
import minirag.models.EmbedResponse

// ── Клиент Ollama: эмбеддинги, декомпозиция вопроса, финальный ответ ─────
class OllamaClient(private val client: HttpClient) {

    suspend fun embed(texts: List<String>): List<List<Double>> {
        val response: EmbedResponse = client.post("$OLLAMA_URL/api/embed") {
            timeout {
                requestTimeoutMillis = 10 * 60 * 1000L
            }
            contentType(ContentType.Application.Json)
            setBody(
                EmbedRequest(
                    model = EMBED_MODEL,
                    input = texts
                )
            )
        }
            .body()
        return response.embeddings
    }

    suspend fun embedAll(
        texts: List<String>,
        batchSize: Int = 16
    ): List<List<Double>> {

        val batches = texts.chunked(batchSize)
        val result = mutableListOf<List<Double>>()

        for ((index, batch) in batches.withIndex()) {
            println(
                "      embedding ${index + 1}/${batches.size} " +
                        "(${batch.size} chunks)"
            )
            result += embed(batch)
        }
        return result
    }

    suspend fun decomposeQuestion(
        question: String
    ): List<String> {

        val prompt = """
            Разбей вопрос пользователя на независимые поисковые запросы.

            Каждый запрос должен быть самостоятельным и понятным
            без дополнительного контекста.

            Не используй фразы:
            - "данное произведение"
            - "этот текст"
            - "в документе"

            Вместо этого используй конкретное название произведения,
            если оно известно из вопроса.

            Если вопрос требует одного факта,
            верни один поисковый запрос.

            Верни ТОЛЬКО JSON-массив строк.
            Не используй Markdown и ```.

            Вопрос:
            $question
        """.trimIndent()

        val response: ChatResponseMessage =
            client.post("$OLLAMA_URL/api/chat") {

                contentType(ContentType.Application.Json)

                setBody(
                    ChatRequest(
                        model = CHAT_MODEL,
                        messages = listOf(
                            ChatMessage(
                                role = "user",
                                content = prompt
                            )
                        ),
                        stream = false,
                        options = ChatOptions(temperature = 0.0)
                    )
                )
            }
                .body()

        val raw = response.message.content

        println("\n[decomposition raw]")
        println(raw)

        return parseQueries(
            raw = raw,
            question = question
        )
    }

    fun parseQueries(
        raw: String,
        question: String
    ): List<String> {
        return try {
            val queries = Json.decodeFromString<List<String>>(
                raw.cleanJsonResponse()
            )
                .map { it.trim() }
                .filter { it.isNotEmpty() }

            queries.ifEmpty { listOf(question) }
        } catch (_: Exception) {
            println("[decompose] invalid JSON, fallback to original question")
            listOf(question)
        }
    }

//    suspend fun testDirectToolResult() {
//        val toolResult = """
//        Возбудитель болезни: Phytophthora infestans
//
//        Симптомы:
//        Первые признаки болезни проявляются в виде сгибания черешка
//        пораженных листьев книзу. Повреждения на листьях и стебле имеют
//        вид крупных зеленоватых, насыщенных влагой пятен неправильной формы.
//        Эти пятна увеличиваются в размерах, приобретают коричневую окраску
//        и становятся бумажистыми.
//
//        В сырую погоду на нижней стороне листа может появляться белый налет
//        спороношения гриба.
//
//        Меры борьбы:
//        Наиболее эффективным средством борьбы с данной болезнью является
//        опрыскивание фунгицидами и наличие системы прогнозирования вспышек
//        заболевания.
//
//        Не следует выращивать томаты на участках, ранее занятых картофелем,
//        или рядом с картофельным полем.
//    """.trimIndent()
//
//        val json = """
//        {
//          "model": "$CHAT_MODEL",
//          "messages": [
//            {
//              "role": "system",
//              "content": "Ты агро-помощник. Отвечай только на основании результата инструмента. Не добавляй сведения из собственных знаний."
//            },
//            {
//              "role": "user",
//              "content": "Какие основные симптомы и меры борьбы с фитофторозом томата?"
//            },
//            {
//              "role": "assistant",
//              "content": "",
//              "tool_calls": [
//                {
//                  "function": {
//                    "name": "searchKnowledge",
//                    "arguments": {
//                      "query": "симптомы и меры борьбы с фитофторозом томата"
//                    }
//                  }
//                }
//              ]
//            },
//            {
//              "role": "tool",
//              "content": ${kotlinx.serialization.json.Json.encodeToString(toolResult)}
//            }
//          ],
//          "stream": false,
//          "options": {
//            "temperature": 0.0
//          }
//        }
//    """.trimIndent()
//
//        println("\n========== DIRECT OLLAMA REQUEST ==========")
//        println(json)
//        println("===========================================\n")
//
//        val response = client.post("$OLLAMA_URL/api/chat") {
//            contentType(ContentType.Application.Json)
//            setBody(json)
//        }
//
//        val responseText = response.bodyAsText()
//
//        println("\n========== DIRECT OLLAMA RESPONSE ==========")
//        println(responseText)
//        println("=============================================")
//    }

    fun close() {
        client.close()
    }
}
