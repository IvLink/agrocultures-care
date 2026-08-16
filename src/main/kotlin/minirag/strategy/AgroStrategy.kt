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
