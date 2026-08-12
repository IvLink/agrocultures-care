package minirag.models

import kotlinx.serialization.Serializable

@Serializable
data class EmbedResponse(val embeddings: List<List<Double>>)

@Serializable
data class ChatMessage(val role: String, val content: String)

@Serializable
data class ChatResponseMessage(val message: ChatMessage)

@Serializable
data class EmbedRequest(val model: String, val input: List<String>)

@Serializable
data class ChatRequest(
    val model: String,
    val messages: List<ChatMessage>,
    val stream: Boolean,
    val options: ChatOptions,
)

@Serializable
data class ChatOptions(val temperature: Double)
