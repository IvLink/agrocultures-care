package minirag

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.http.client.HttpClientFactoryResolver
import ai.koog.prompt.executor.llms.MultiLLMPromptExecutor
import ai.koog.prompt.llm.LLMCapability
import ai.koog.prompt.llm.LLMProvider
import ai.koog.prompt.llm.LLModel
import minirag.agents.AgroKnowledgeTool
import minirag.bge_reranker.BgeReranker
import minirag.config.createHttpClient
import minirag.eval.GroundednessJudge
import minirag.network.LoggingKoogHttpClientFactory
import minirag.ollama.OllamaClient
import minirag.pdf.readPdf
import minirag.retrieval.Retriever
import minirag.strategy.agroStrategy
import minirag.eval.runAgentForEval
import minirag.text.splitIntoChunks
import ai.koog.prompt.executor.ollama.client.OllamaClient as OllamaClientKoog

val gemma4 = LLModel(
    provider = LLMProvider.Ollama,
    id = "gemma4:latest",
    capabilities = listOf(
        LLMCapability.Temperature,
        LLMCapability.Schema.JSON.Basic,
        LLMCapability.Tools
    ),
    contextLength = 131072 // ориентировочно; точное значение — `ollama show gemma4:latest`
)

suspend fun main() {
    val client = createHttpClient()
    val ollama = OllamaClient(client)

    val retriever = Retriever(ollama)
    val reranker = BgeReranker(
        modelPath =
            "src/main/kotlin/minirag/models/bge_reranker_v2_m3/onnx/model.onnx",
        tokenizerPath =
            "src/main/kotlin/minirag/models/bge_reranker_v2_m3/tokenizer.json"
    )

    print("Путь к PDF-файлу: ")
    val pdfPath = readlnOrNull().orEmpty()
        .trim()
        .trim('"')

//    print("Твой вопрос к документу: ")
//    val question = readlnOrNull().orEmpty()
//        .trim()

    println("[1/4] читаю PDF...")
    val text = readPdf(pdfPath)
    println("      символов: ${text.length}")

    println("[2/4] режу на чанки...")
    val chunks = splitIntoChunks(text)
    println("      чанков: ${chunks.size}")

    println("[3/4] считаю эмбеддинги...")
    val chunkVecs = ollama.embedAll(chunks)

    println(
        "      векторов: ${chunkVecs.size}, " +
                "размерность: ${chunkVecs.firstOrNull()?.size}"
    )
    val agroTool = AgroKnowledgeTool(
        retriever = retriever,
        reranker = reranker,
        chunks = chunks,
        chunkVecs = chunkVecs
    )

//    val testTool = TestTool()
//
//    val agent = AIAgent(
//        promptExecutor = MultiLLMPromptExecutor(OllamaClientKoog()),
//        llmModel = gemma4,
//        systemPrompt = """
//        Ты тестовый ассистент.
//
//        Если для ответа нужна информация из инструмента,
//        используй testTool.
//
//        После получения результата инструмента
//        используй его при формировании ответа.
//    """.trimIndent(),
//        toolRegistry = ToolRegistry {
//            tools(testTool)
//        },
//        strategy = agroStrategy
//    )

    @Suppress("UnstableApiUsage")
    val defaultFactory = HttpClientFactoryResolver.resolve()

    val loggingFactory = LoggingKoogHttpClientFactory(
        defaultFactory
    )

    @Suppress("UnstableApiUsage")
    val ollamaKoog = OllamaClientKoog(httpClientFactory = loggingFactory)

    val agent = AIAgent(
        promptExecutor = MultiLLMPromptExecutor(ollamaKoog),
        llmModel = gemma4,
        systemPrompt = """
    Ты агро-помощник, работающий с локальной агрономической базой знаний.

    ОБЯЗАТЕЛЬНО используй searchKnowledge, если вопрос относится к:
    - болезням растений;
    - симптомам заболеваний;
    - возбудителям заболеваний;
    - условиям развития заболеваний;
    - мерам борьбы;
    - лечению;
    - профилактике;
    - любой информации, которая может находиться в агрономическом документе.

    Если вопрос относится к агрономической информации, сначала вызови
    searchKnowledge и только после получения результата сформируй ответ.

    Не отвечай на такой вопрос из собственных знаний.

    Не придумывай сведения, которых нет в результате searchKnowledge.
""".trimIndent(),
        toolRegistry = ToolRegistry {
            tools(listOf(agroTool))
        },
        strategy = agroStrategy
    )

    val question = "Какие меры борьбы с фитофторозом указаны в документе?"

    val evalRun = runAgentForEval(
        agent = agent,
        agroTool = agroTool,
        question = question
    )

    val groundednessJudge = GroundednessJudge(
        promptExecutor = MultiLLMPromptExecutor(ollamaKoog),
        model = gemma4
    )

    val groundedness = groundednessJudge.evaluate(
        question = question,
        context = evalRun.retrievedContext,
        answer = evalRun.answer
    )

    println()
    println("========== TOOL ==========")
    println("searchKnowledge called: ${evalRun.toolCalled}")

    println()
    println("========== RETRIEVED CONTEXT ==========")
    println(evalRun.retrievedContext ?: "<NO TOOL RESULT>")

    println()
    println("========== FINAL ANSWER ==========")
    println(evalRun.answer)

    println()
    println("========== GROUNDEDNESS ==========")
    println("grounded: ${groundedness.grounded}")

    groundedness.unsupportedClaims.forEach {
        println("- $it")
    }

    println("==================================")

//    val result = agent.run(
//        "Используй testTool и скажи, какое число он вернул."
//    )
//
//    println("\n========== FINAL ANSWER ==========")
//    println(result)

    agent.close()
    reranker.close()
    ollama.close()
}
