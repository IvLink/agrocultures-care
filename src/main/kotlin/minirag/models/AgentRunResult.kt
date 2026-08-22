package minirag.models

data class AgentRunResult(
    val answer: String,
    val retrievedContext: String?,
    val toolCalled: Boolean,
    val webSearchCalled: Boolean
)