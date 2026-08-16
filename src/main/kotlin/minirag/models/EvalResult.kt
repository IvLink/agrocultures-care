package minirag.models

data class EvalResult(
    val id: String,
    val question: String,
    val retrievedContext: String,
    val answer: String,
    val grounded: Boolean,
    val unsupportedClaims: List<String>
)