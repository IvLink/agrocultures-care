package minirag.eval

import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.llms.MultiLLMPromptExecutor
import ai.koog.prompt.llm.LLModel
import kotlinx.serialization.json.Json
import minirag.models.GroundednessJudgeResult

class GroundednessJudge(
    private val promptExecutor: MultiLLMPromptExecutor,
    private val model: LLModel
) {
    private val json = Json {
        ignoreUnknownKeys = true
    }

    suspend fun evaluate(
        question: String,
        context: String?,
        answer: String
    ): GroundednessJudgeResult {

        if (context.isNullOrBlank()) {
            return GroundednessJudgeResult(
                grounded = false,
                unsupportedClaims = listOf(
                    "Ответ сформирован без результата searchKnowledge."
                )
            )
        }

        val evaluationPrompt = prompt("groundedness-eval") {
            system(
                """
                Ты проверяешь, основан ли ответ исключительно на предоставленном контексте.

                Правила:
                1. Рассматривай каждое существенное утверждение в ANSWER отдельно.
                2. Утверждение считается подтвержденным только если оно
                   непосредственно содержится или однозначно следует из CONTEXT.
                3. Нельзя использовать собственные знания.
                4. Если хотя бы одно существенное утверждение не подтверждается
                   CONTEXT, grounded должен быть false.
                5. Верни ТОЛЬКО JSON без markdown и без дополнительного текста.
                6. Проверяй также, отвечает ли ANSWER именно на QUESTION.
                7. Нельзя переносить сведения о другом заболевании на заболевание,
                    указанное в QUESTION.

                Формат:

                {
                  "grounded": true,
                  "unsupportedClaims": []
                }

                или:

                {
                  "grounded": false,
                  "unsupportedClaims": [
                    "неподтвержденное утверждение"
                  ]
                }
                """.trimIndent()
            )

            user(
                """
                    QUESTION:
                    $question
                
                    CONTEXT:
                    $context
                
                    ANSWER:
                    $answer
                    """.trimIndent()
            )
        }

        val response = promptExecutor.execute(
            prompt = evaluationPrompt,
            model = model
        )

        val text = response
            .textContent()
            .trim()

        val jsonExtracted = extractJson(text)

        return json.decodeFromString<GroundednessJudgeResult>(jsonExtracted)
    }

    private fun extractJson(raw: String): String {
        val text = raw.trim()

        if (text.startsWith("```")) {
            return text
                .replaceFirst(Regex("^```(?:json)?\\s*"), "")
                .replaceFirst(Regex("\\s*```$"), "")
                .trim()
        }

        return text
    }
}