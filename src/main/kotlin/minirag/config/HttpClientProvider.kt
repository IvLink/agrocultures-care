package minirag.config

import io.ktor.client.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.serialization.json.Json

fun createHttpClient(): HttpClient = HttpClient(CIO) {
    install(ContentNegotiation) {
        json(
            Json {
                ignoreUnknownKeys = true
                // не слать "field": null для необязательных полей
                // (напр. YandexSearchRequest.region) — некоторые API
                // трактуют явный null иначе, чем отсутствие поля.
                explicitNulls = false
            }
        )
    }
    install(HttpTimeout) {
        requestTimeoutMillis = 10 * 60 * 1000L
        connectTimeoutMillis = 3_000L
        socketTimeoutMillis = 10 * 60 * 1000L
    }
}
