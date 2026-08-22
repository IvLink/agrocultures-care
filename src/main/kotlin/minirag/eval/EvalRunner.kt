package minirag.eval

import ai.koog.agents.core.agent.AIAgent
import minirag.agents.AgroKnowledgeTool
import minirag.agents.WebSearchTool
import minirag.models.AgentRunResult

suspend fun runAgentForEval(
    agent: AIAgent<String, String>,
    agroTool: AgroKnowledgeTool?,
    webSearchTool: WebSearchTool?,
    question: String
): AgentRunResult {

    agroTool?.clearLastResult()
    webSearchTool?.clearLastResult()

    val answer = agent.run(question)

    return AgentRunResult(
        answer = answer,
        retrievedContext = combineContext(agroTool?.lastResult, webSearchTool?.lastResult),
        toolCalled = agroTool?.lastResult != null,
        webSearchCalled = webSearchTool?.lastResult != null
    )
}

/*
 * GroundednessJudge должен проверять ANSWER против результата ЛЮБОГО
 * вызванного тула, не только searchKnowledge — иначе вопрос вида
 * Test 2/3 (только webSearch) всегда получал бы пустой context и
 * ложно помечался как ungrounded, даже если агент честно ответил
 * по данным webSearch (или честно сказал, что данных нет).
 */
private fun combineContext(agroResult: String?, webResult: String?): String? {
    val parts = buildList {
        agroResult?.let { add("=== searchKnowledge ===\n$it") }
        webResult?.let { add("=== webSearch ===\n$it") }
    }
    return parts.takeIf { it.isNotEmpty() }?.joinToString("\n\n")
}