package minirag.network

import ai.koog.http.client.KoogHttpClient
import kotlinx.coroutines.flow.Flow
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.reflect.KClass

@Suppress("UnstableApiUsage")
class LoggingKoogHttpClientFactory(
    private val delegate: KoogHttpClient.Factory
) : KoogHttpClient.Factory {

    override fun create(
        clientName: String,
        baseUrl: String,
        headers: Map<String, String>,
        queryParameters: Map<String, String>,
        requestTimeoutMillis: Long,
        connectTimeoutMillis: Long,
        socketTimeoutMillis: Long,
        json: Json
    ): KoogHttpClient {

        val client = delegate.create(
            clientName = clientName,
            baseUrl = baseUrl,
            headers = headers,
            queryParameters = queryParameters,
            requestTimeoutMillis = requestTimeoutMillis,
            connectTimeoutMillis = connectTimeoutMillis,
            socketTimeoutMillis = socketTimeoutMillis,
            json = json
        )

        return LoggingKoogHttpClient(client)
    }
}

@Suppress("UnstableApiUsage")
private class LoggingKoogHttpClient(
    private val delegate: KoogHttpClient
) : KoogHttpClient {

    override val clientName: String
        get() = delegate.clientName

    override suspend fun <R : Any> get(
        path: String,
        responseType: KClass<R>,
        parameters: Map<String, String>,
        headers: Map<String, String>
    ): R {
        return delegate.get(
            path = path,
            responseType = responseType,
            parameters = parameters,
            headers = headers
        )
    }

    override suspend fun <T : Any, R : Any> post(
        path: String,
        requestBody: T,
        requestBodyType: KClass<T>,
        responseType: KClass<R>,
        parameters: Map<String, String>,
        headers: Map<String, String>
    ): R {
        if (path == "api/chat" && requestBody is String) {
            val fixedRequestBody = removeEmptyUserMessage(requestBody)

            println()
            println("========== KOOG → OLLAMA REQUEST ==========")
            println(fixedRequestBody)
            println("============================================")
            println()

            return delegate.post(
                path = path,
                requestBody = fixedRequestBody,
                requestBodyType = String::class,
                responseType = responseType,
                parameters = parameters,
                headers = headers
            )
        }

        return delegate.post(
            path = path,
            requestBody = requestBody,
            requestBodyType = requestBodyType,
            responseType = responseType,
            parameters = parameters,
            headers = headers
        )
    }

    override fun <T : Any, R : Any, O : Any> sse(
        path: String,
        requestBody: T,
        requestBodyType: KClass<T>,
        dataFilter: (String?) -> Boolean,
        decodeStreamingResponse: (String) -> R,
        processStreamingChunk: (R) -> O?,
        parameters: Map<String, String>,
        headers: Map<String, String>
    ): Flow<O> {
        return delegate.sse(
            path = path,
            requestBody = requestBody,
            requestBodyType = requestBodyType,
            dataFilter = dataFilter,
            decodeStreamingResponse = decodeStreamingResponse,
            processStreamingChunk = processStreamingChunk,
            parameters = parameters,
            headers = headers
        )
    }

    override fun <T : Any> lines(
        path: String,
        requestBody: T,
        requestBodyType: KClass<T>,
        parameters: Map<String, String>,
        headers: Map<String, String>
    ): Flow<String> {
        return delegate.lines(
            path = path,
            requestBody = requestBody,
            requestBodyType = requestBodyType,
            parameters = parameters,
            headers = headers
        )
    }

    override fun close() {
        delegate.close()
    }
}


private fun removeEmptyUserMessage(json: String): String {
    val root = Json.parseToJsonElement(json).jsonObject

    val messages = root["messages"]?.jsonArray
        ?: return json

    val filtered = messages.filterNot { message ->
        val obj = message.jsonObject

        obj["role"]?.jsonPrimitive?.content == "user" &&
                obj["content"]?.jsonPrimitive?.content == ""
    }

    if (filtered.size == messages.size) {
        return json
    }

    val newRoot = JsonObject(
        root.toMutableMap()
            .apply {
                this["messages"] = kotlinx.serialization.json.JsonArray(filtered)
            }
    )

    return Json.encodeToString(
        JsonObject.serializer(),
        newRoot
    )
}

/**
 * Workaround for a koog-agents bug (present through at least 1.1.1): when converting a
 * `Message.User` that only wraps a `MessagePart.Tool.Result` (i.e. a tool result, which Koog
 * encodes as a User message under the hood), `OllamaConverters.toOllamaChatMessages` still emits
 * a spurious `{"role":"user","content":""}` right before the real `{"role":"tool",...}` message.
 * Ollama-served small models (e.g. gemma here) sometimes misread that empty turn as "the tool
 * hasn't answered yet" and stall instead of using the tool result.
 *
 * The DTOs are `internal` to the koog-agents module, so we can't fix the converter directly;
 * instead we reflectively drop those empty/content-less "user" messages from the request just
 * before it goes over the wire, and rebuild the request via its own `copy()`.
 * One thing to note: this only patches the non-streaming post() path,
 * which is what you're using ("stream":false in your logs).
 * If you ever switch the agent to streaming, the same fix would need to be mirrored in sse()/lines().
 */
//@Suppress("UNCHECKED_CAST")
//private fun <T : Any> stripEmptyToolResultUserTurns(requestBody: T): T {
//    val kClass = requestBody::class
//    val messagesProperty = kClass.memberProperties.find { it.name == "messages" } ?: return requestBody
//    messagesProperty.isAccessible = true
//    val messages = messagesProperty.call(requestBody) as? List<*> ?: return requestBody
//
//    fun isEmptyUserTurn(message: Any?): Boolean {
//        if (message == null) return false
//        val messageProperties = message::class.memberProperties
//        val role = messageProperties.find { it.name == "role" }
//            ?.apply { isAccessible = true }?.call(message) as? String
//        val content = messageProperties.find { it.name == "content" }
//            ?.apply { isAccessible = true }?.call(message) as? String
//        val images = messageProperties.find { it.name == "images" }
//            ?.apply { isAccessible = true }?.call(message) as? List<*>
//        return role == "user" && content.isNullOrBlank() && images.isNullOrEmpty()
//    }
//
//    val filteredMessages = messages.filterNot(::isEmptyUserTurn)
//    if (filteredMessages.size == messages.size) return requestBody
//
//    val dropped = messages.size - filteredMessages.size
//    println("[LoggingKoogHttpClientFactory] dropped $dropped empty tool-result-artifact 'user' message(s) before sending to Ollama")
//
//    val copyFunction = kClass.memberFunctions.find { it.name == "copy" } ?: return requestBody
//    copyFunction.isAccessible = true
//    val instanceParameter = copyFunction.instanceParameter ?: return requestBody
//    val messagesParameter = copyFunction.parameters.find { it.name == "messages" } ?: return requestBody
//
//    return copyFunction.callBy(
//        mapOf(
//            instanceParameter to requestBody,
//            messagesParameter to filteredMessages
//        )
//    ) as T
//}