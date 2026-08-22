package minirag.models

import kotlinx.serialization.SerialName
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
data class ChatOptions(
    val temperature: Double,
    @SerialName("num_ctx") val numCtx: Long? = null
)

@Serializable
data class AgroSearchArgs(val query: String)

@Serializable
data class GroundednessJudgeResult(
    val grounded: Boolean,
    val unsupportedClaims: List<String>
)

@Serializable
data class DomainClassification(
    val agronomic: Boolean,
    val needsDocument: Boolean = true
)
@Serializable
data class WebSearchArgs(
    val query: String,
    val recency: String? = null
)