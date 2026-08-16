package minirag.eval

import ai.koog.agents.core.agent.AIAgent
import minirag.agents.AgroKnowledgeTool
import minirag.models.AgentRunResult

suspend fun runAgentForEval(
    agent: AIAgent<String, String>,
    agroTool: AgroKnowledgeTool,
    question: String
): AgentRunResult {

    agroTool.clearLastResult()

    val answer = agent.run(question)

    return AgentRunResult(
        answer = answer,
        retrievedContext = agroTool.lastResult,
        toolCalled = agroTool.lastResult != null
    )
}