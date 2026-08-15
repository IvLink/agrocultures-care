package minirag.agents

import ai.koog.agents.core.tools.annotations.LLMDescription
import ai.koog.agents.core.tools.annotations.Tool
import ai.koog.agents.core.tools.reflect.ToolSet

@LLMDescription("Тестовый инструмент для проверки передачи результата Tool обратно в LLM")
class TestTool : ToolSet {

    @Tool
    @LLMDescription("Возвращает фиксированный тестовый результат")
    fun testTool(): String {
        println("========== TEST TOOL EXECUTED ==========")

        return "Тестовый результат: число 42."
    }
}