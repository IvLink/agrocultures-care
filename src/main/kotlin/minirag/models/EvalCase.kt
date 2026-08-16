package minirag.models

data class EvalCase(
    val id: String,
    val question: String,
    val expectedAnswerFacts: List<String>
)
