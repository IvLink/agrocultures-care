package minirag.strategy

import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.core.dsl.extension.nodeExecuteTools
import ai.koog.agents.core.dsl.extension.nodeLLMRequest
import ai.koog.agents.core.dsl.extension.nodeLLMSendToolResults
import ai.koog.agents.core.dsl.extension.onTextMessage
import ai.koog.agents.core.dsl.extension.onToolCalls

val agroStrategy = strategy<String, String>("agro-rag") {

    val requestLLM by nodeLLMRequest()
    val executeTools by nodeExecuteTools()
    val sendToolResults by nodeLLMSendToolResults()

    edge(nodeStart forwardTo requestLLM)
    edge(requestLLM forwardTo nodeFinish onTextMessage { true })
    edge(requestLLM forwardTo executeTools onToolCalls { true })
    edge(executeTools forwardTo sendToolResults)
    edge(sendToolResults forwardTo nodeFinish onTextMessage { true })
    edge(sendToolResults forwardTo executeTools onToolCalls { true })
}

//enum class UserIntent { TECHNICAL, BILLING, UNKNOWN }
//
//val intentClassifierNode by node<String, UserIntent>("intent_classifier") { userInput ->
//    // Доступ к LLM-клиенту из контекста или DI
//    val llmClient = context.get<LlmClient>()
//
//    // Открываем изменяемую сессию для этой ноды
//    val response = llmClient.writeSession {
//        // 1. Задаем системные инструкции (Промпт)
//        system("""
//            Ты — классификатор запросов поддержки.
//            Определи категорию сообщения пользователя.
//            Возвращай ТОЛЬКО одно слово из списка: TECHNICAL, BILLING, UNKNOWN.
//            Никакого лишнего текста.
//        """.trimIndent())
//
//        // 2. Добавляем текущий ввод пользователя в историю сессии
//        user(userInput)
//
//        // 3. Вызываем модель (например, GPT-4o или локальную через Ollama)
//        generate()
//    }
//
//    // Шаг 3. Парсим текстовый ответ модели в строго типизированный Enum Kotlin
//    val modelTextAnswer = response.text.trim().uppercase()
//
//    runCatching {
//        UserIntent.valueOf(modelTextAnswer)
//    }.getOrElse {
//        UserIntent.UNKNOWN
//    }
//}