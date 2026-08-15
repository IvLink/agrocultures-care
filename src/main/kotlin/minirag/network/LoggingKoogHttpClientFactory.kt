package minirag.network

import ai.koog.http.client.KoogHttpClient
import ai.koog.http.client.KoogHttpClientException
import kotlinx.coroutines.flow.Flow
import kotlinx.serialization.json.Json
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

        if (path == "api/chat") {
            println()
            println("========== KOOG → OLLAMA REQUEST ==========")
            println(requestBody)
            println("============================================")
            println()
        }

        val response = delegate.post(
            path = path,
            requestBody = requestBody,
            requestBodyType = requestBodyType,
            responseType = responseType,
            parameters = parameters,
            headers = headers
        )

        if (path == "api/chat") {
            println()
            println("========== OLLAMA ← KOOG RESPONSE ==========")
            println(response)
            println("=============================================")
            println()
        }

        return response
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