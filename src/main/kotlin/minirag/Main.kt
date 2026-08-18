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
import minirag.eval.calibrateThreshold
import minirag.network.LoggingKoogHttpClientFactory
import minirag.ollama.OllamaClient
import minirag.pdf.readPdfPages
import minirag.retrieval.LexicalRetriever
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

    print("Твой вопрос к документу: ")
    val question = readlnOrNull().orEmpty()
        .trim()

    println("[1/4] читаю PDF...")
    val pages = readPdfPages(pdfPath)
    println("      страниц: ${pages.size}")

    println("[2/4] режу на чанки...")
    val chunks = splitIntoChunks(pages)
    println("      чанков: ${chunks.size}")

    println("[3/4] считаю эмбеддинги...")
    val chunkVecs = ollama.embedAll(
        chunks.map { it.retrievalText() }
    )

    println(
        "      векторов: ${chunkVecs.size}, " +
                "размерность: ${chunkVecs.firstOrNull()?.size}"
    )

    val lexicalRetriever = LexicalRetriever(chunks)

    println("[4/4] калибрую relevance threshold по eval-набору...")
    val calibration = calibrateThreshold(
        ollama = ollama,
        retriever = retriever,
        lexicalRetriever = lexicalRetriever,
        reranker = reranker,
        chunks = chunks,
        chunkVecs = chunkVecs
    )

    val agroTool = AgroKnowledgeTool(
        ollama = ollama,
        retriever = retriever,
        lexicalRetriever = lexicalRetriever,
        reranker = reranker,
        chunks = chunks,
        chunkVecs = chunkVecs,
        relevanceThreshold = calibration.threshold
    )

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
        temperature = 0.0,
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

    Используй ТОЛЬКО информацию, содержащуюся в результате searchKnowledge.

    Не используй собственные знания для дополнения ответа.

    Не смешивай информацию из разных заболеваний, вредителей,
    культур или разделов документа.

    Если найденный контекст относится к другому заболеванию или
    другой теме, не используй его для ответа.

    Если в результате searchKnowledge нет достаточной информации
    для ответа, прямо сообщи, что в найденном контексте
    недостаточно информации.

    Не придумывай сведения, которых нет в результате searchKnowledge.
""".trimIndent(),
        toolRegistry = ToolRegistry {
            tools(listOf(agroTool))
        },
        strategy = agroStrategy,
    )

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

    agent.close()
    reranker.close()
    ollama.close()
}
