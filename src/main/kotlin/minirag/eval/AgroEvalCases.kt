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