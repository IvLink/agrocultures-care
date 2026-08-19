package minirag

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.http.client.HttpClientFactoryResolver
import ai.koog.prompt.executor.llms.MultiLLMPromptExecutor
import ai.koog.prompt.executor.ollama.client.ContextWindowStrategy
import ai.koog.prompt.llm.LLMCapability
import ai.koog.prompt.llm.LLMProvider
import ai.koog.prompt.llm.LLModel
import minirag.agents.AgroKnowledgeTool
import minirag.reranker.BgeReranker
import minirag.config.CHAT_MODEL
import minirag.config.OLLAMA_NUM_CTX
import minirag.config.RELEVANCE_THRESHOLD
import minirag.config.createHttpClient
import minirag.eval.GroundednessJudge
import minirag.eval.calibrateThreshold
import minirag.models.AgentRunResult
import minirag.models.GroundednessJudgeResult
import minirag.network.LoggingKoogHttpClientFactory
import minirag.ollama.OllamaClient
import minirag.pdf.readPdfPages
import minirag.retrieval.LexicalRetriever
import minirag.retrieval.Retriever
import minirag.strategy.agroStrategy
import minirag.eval.runAgentForEval
import minirag.text.splitIntoChunks
import ai.koog.prompt.executor.ollama.client.OllamaClient as OllamaClientKoog
import kotlin.system.measureTimeMillis

val gemma4 = LLModel(
    provider = LLMProvider.Ollama,
    id = CHAT_MODEL,
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
        modelPath = "models/bge_reranker_v2_m3/onnx/model.onnx",
        tokenizerPath = "models/bge_reranker_v2_m3/tokenizer.json"
    )

    print("Путь к PDF-файлу: ")
    val pdfPath = readlnOrNull().orEmpty()
        .trim()
        .trim('"')

    print("Твой вопрос к документу: ")
    val question = readlnOrNull().orEmpty()
        .trim()

    /*
     * Domain gate: отдельный дешёвый классификационный запрос ДО
     * запуска дорогого пайплайна (PDF -> чанки -> эмбеддинги ->
     * калибровка -> agent), а не системный промпт агента как
     * единственная защита — с локальной моделью это лишь мягкая
     * рекомендация, а не гарантия, что tool-calling агент сам решит
     * не звать searchKnowledge на нерелевантный вопрос.
     */
    val isAgronomic: Boolean
    val gateElapsedMs = measureTimeMillis {
        isAgronomic = ollama.classifyAgronomic(question)
    }
    println("[domain gate] agronomic=$isAgronomic (${gateElapsedMs}мс)")

    if (!isAgronomic) {
        println(
            "Вопрос не относится к агрономической базе знаний — " +
                    "пропускаю PDF/эмбеддинги/агент."
        )
        reranker.close()
        ollama.close()
        return
    }

    println("[1/4] читаю PDF...")
    val pages = readPdfPages(pdfPath)
    println("      страниц: ${pages.size}")

    println("[2/4] режу на чанки...")
    val chunks = splitIntoChunks(pages)
    println("      чанков: ${chunks.size}")

    var chunkVecs: List<List<Double>> = emptyList()
    val embeddingsMs = measureTimeMillis {
        println("[3/4] считаю эмбеддинги...")
        chunkVecs = ollama.embedAll(
            chunks.map { it.retrievalText() }
        )
    }

    println(
        "      векторов: ${chunkVecs.size}, " +
                "размерность: ${chunkVecs.firstOrNull()?.size}"
    )

    val lexicalRetriever = LexicalRetriever(chunks)

    /*
     * calibrateThreshold — это верификация/подбор RELEVANCE_THRESHOLD
     * (см. minirag.eval.RerankerCalibration), а не часть прод-ответа:
     * она прогоняет ПОЛНЫЙ pipeline (decompose -> retrieval -> rerank)
     * ещё раз для 7 eval-запросов, что на практике оказывается дороже
     * самого ответа на вопрос пользователя. SKIP_CALIBRATION позволяет
     * пропустить эту проверку и сразу использовать текущий откалиброванный
     * RELEVANCE_THRESHOLD из AppConfig.kt — на прод-ответ это не влияет,
     * влияет только на то, проверяем ли мы актуальность порога здесь и сейчас.
     */
    val skipCalibration = System.getenv("SKIP_CALIBRATION") == "1"

    var threshold = RELEVANCE_THRESHOLD
    val calibrationMs = measureTimeMillis {
        if (skipCalibration) {
            println("[4/4] калибровка пропущена (SKIP_CALIBRATION=1), " +
                    "используется RELEVANCE_THRESHOLD=$RELEVANCE_THRESHOLD")
        } else {
            println("[4/4] калибрую relevance threshold по eval-набору...")
            threshold = calibrateThreshold(
                ollama = ollama,
                retriever = retriever,
                lexicalRetriever = lexicalRetriever,
                reranker = reranker,
                chunks = chunks,
                chunkVecs = chunkVecs
            ).threshold
        }
    }

    val agroTool = AgroKnowledgeTool(
        ollama = ollama,
        retriever = retriever,
        lexicalRetriever = lexicalRetriever,
        reranker = reranker,
        chunks = chunks,
        chunkVecs = chunkVecs,
        relevanceThreshold = threshold
    )

    @Suppress("UnstableApiUsage")
    val defaultFactory = HttpClientFactoryResolver.resolve()

    val loggingFactory = LoggingKoogHttpClientFactory(
        defaultFactory
    )

    @Suppress("UnstableApiUsage")
    val ollamaKoog = OllamaClientKoog(
        httpClientFactory = loggingFactory,
        /*
         * Без явного num_ctx Ollama сама выбирает размер контекстного
         * окна (см. AppConfig.OLLAMA_NUM_CTX) — на этой машине это
         * заставляло ~72% слоёв модели уходить на CPU, потому что
         * KV-cache под такой контекст не помещался в VRAM видеокарты.
         */
        contextWindowStrategy = ContextWindowStrategy.Companion.Fixed(OLLAMA_NUM_CTX)
    )

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

    lateinit var evalRun: AgentRunResult
    lateinit var groundedness: GroundednessJudgeResult

    val agentMs = measureTimeMillis {
        evalRun = runAgentForEval(
            agent = agent,
            agroTool = agroTool,
            question = question
        )
    }

    val groundednessMs = measureTimeMillis {
        val groundednessJudge = GroundednessJudge(
            promptExecutor = MultiLLMPromptExecutor(ollamaKoog),
            model = gemma4
        )

        groundedness = groundednessJudge.evaluate(
            question = question,
            context = evalRun.retrievedContext,
            answer = evalRun.answer
        )
    }

    val answerMs = agentMs + groundednessMs

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

    println()
    println("========== TIMING ==========")
    println("domain gate: ${gateElapsedMs}мс")
    println("эмбеддинги: ${embeddingsMs}мс")
    println(
        if (skipCalibration) "калибровка: пропущена"
        else "калибровка: ${calibrationMs}мс"
    )
    println("  agent (tool-call + final answer): ${agentMs}мс")
    println("  groundedness judge: ${groundednessMs}мс")
    println("ответ на вопрос: ${answerMs}мс")
    println("=============================")

    agent.close()
    reranker.close()
    ollama.close()
}
