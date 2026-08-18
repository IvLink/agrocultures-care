package minirag.eval

import minirag.models.EvalCase

val evalCases = listOf(
    EvalCase(
        id = "phytophthora-01",
        question = "Какой возбудитель фитофтороза томата?",
        expectedAnswerFacts = listOf(
            "Phytophthora infestans"
        )
    ),

    EvalCase(
        id = "phytophthora-02",
        question = "Какие основные симптомы фитофтороза томата?",
        expectedAnswerFacts = listOf(
            "сгибание черешка",
            "зеленоватые пятна",
            "коричневая окраска",
            "белый налет спороношения"
        )
    ),

    EvalCase(
        id = "phytophthora-03",
        question = "Какие повреждения возникают на плодах при фитофторозе?",
        expectedAnswerFacts = listOf(
            "коричневато-зеленые пятна",
            "шероховатая поверхность",
            "маслянистый вид"
        )
    )
)

/*
 * Заведомо нерелевантные для агрономической базы знаний запросы.
 *
 * Используются RerankerCalibration'ом как отрицательные примеры:
 * лучший найденный кандидат для такого запроса всё равно должен
 * получить низкий score, потому что в документе просто нет ответа.
 */
val negativeEvalQueries = listOf(
    "мнемоника для запоминания названий вредителей",
    "какая столица Франции",
    "рецепт борща",
    "как установить Kotlin"
)