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

        /*
         * Раньше пустой context сразу давал grounded=false — это было
         * верно, пока context мог прийти только от searchKnowledge
         * (не вызван -> точно нечем подкрепить ответ). Теперь context
         * может отсутствовать и в легитимном случае: webSearch честно
         * вернул "результатов нет", а ANSWER это признаёт, ничего не
         * выдумывая — такой ответ обязан считаться grounded. Поэтому
         * решение отдаётся LLM (см. правило 8/9 в системном промпте
         * ниже), а не жёсткому шорткату здесь.
         */
        val effectiveContext = context
            ?.takeIf { it.isNotBlank() }
            ?: "(ни один инструмент не вернул контекст)"

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
                8. Если CONTEXT пуст или указывает, что данных не найдено, а
                   ANSWER честно сообщает об отсутствии данных и не выдумывает
                   конкретные факты (цены, названия препаратов, наличие и т.п.) —
                   это grounded=true: отсутствие фактов не выдано за факт.
                9. Если же в этой ситуации ANSWER всё равно утверждает
                   конкретные факты, не подтверждённые CONTEXT — это
                   grounded=false.

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
                    $effectiveContext

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