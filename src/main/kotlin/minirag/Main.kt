package minirag

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.tools.Tool
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.http.client.HttpClientFactoryResolver
import ai.koog.prompt.executor.llms.MultiLLMPromptExecutor
import ai.koog.prompt.executor.ollama.client.ContextWindowStrategy
import ai.koog.prompt.llm.LLMCapability
import ai.koog.prompt.llm.LLMProvider
import ai.koog.prompt.llm.LLModel
import io.ktor.client.HttpClient
import minirag.agents.AgroKnowledgeTool
import minirag.agents.WebSearchTool
import minirag.config.CHAT_MODEL
import minirag.config.OLLAMA_NUM_CTX
import minirag.config.RELEVANCE_THRESHOLD
import minirag.config.SERPER_API_KEY_ENV
import minirag.config.SERPER_DEFAULT_GL
import minirag.config.SERPER_DEFAULT_HL
import minirag.config.SERPER_GL_ENV
import minirag.config.SERPER_HL_ENV
import minirag.config.createHttpClient
import minirag.eval.GroundednessJudge
import minirag.eval.calibrateThreshold
import minirag.eval.runAgentForEval
import minirag.models.AgentRunResult
import minirag.models.DocumentChunk
import minirag.models.GroundednessJudgeResult
import minirag.network.LoggingKoogHttpClientFactory
import minirag.ollama.OllamaClient
import minirag.pdf.readPdfPages
import minirag.reranker.BgeReranker
import minirag.retrieval.LexicalRetriever
import minirag.retrieval.Retriever
import minirag.strategy.agroStrategy
import minirag.text.splitIntoChunks
import minirag.web.SerperSearchClient
import minirag.web.WebSearchClient
import ai.koog.prompt.executor.ollama.client.OllamaClient as OllamaClientKoog

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

private data class TimingReport(
    val gateMs: Long,
    val documentSkipped: Boolean,
    val embeddingsMs: Long,
    val calibrationMs: Long,
    val calibrationSkipped: Boolean,
    val agentMs: Long,
    val groundednessMs: Long,
    val groundednessSkipped: Boolean
) {
    val answerMs get() = agentMs + groundednessMs
}

suspend fun main() {
    val client = createHttpClient()
    val ollama = OllamaClient(client)

    val retriever = Retriever(ollama)
    val reranker = BgeReranker(
        modelPath = "models/bge_reranker_v2_m3/onnx/model.onnx",
        tokenizerPath = "models/bge_reranker_v2_m3/tokenizer.json"
    )

    val pdfPath = promptForPdfPath()
    val question = promptForQuestion()

    val (classification, gateMs) = timed { ollama.classifyAgronomic(question) }
    println(
        "[domain gate] agronomic=${classification.agronomic} " +
                "needsDocument=${classification.needsDocument} (${gateMs}мс)"
    )

    if (!classification.agronomic) {
        println(
            "Вопрос не относится к агрономической базе знаний — " +
                    "пропускаю PDF/эмбеддинги/агент."
        )
        reranker.close()
        ollama.close()
        return
    }

    /*
     * Вопросы только про цену/наличие/покупку (webSearch) не требуют
     * содержимого локального документа — needsDocument=false отсекает
     * самую дорогую часть пайплайна (чтение PDF + эмбеддинги, ~80с)
     * ещё до того, как она запустится, а не после.
     */
    var agroTool: AgroKnowledgeTool? = null
    var embeddingsMs = 0L
    var calibrationMs = 0L
    var calibrationSkipped = true

    if (classification.needsDocument) {
        val chunks = loadDocumentChunks(pdfPath)
        val (chunkVecs, embMs) = timed {
            println("[3/4] считаю эмбеддинги...")
            ollama.embedAll(chunks.map { it.retrievalText() })
        }
        embeddingsMs = embMs
        println(
            "      векторов: ${chunkVecs.size}, " +
                    "размерность: ${chunkVecs.firstOrNull()?.size}"
        )

        val lexicalRetriever = LexicalRetriever(chunks)

        val (threshold, calMs, calSkipped) = resolveRelevanceThreshold(
            ollama = ollama,
            retriever = retriever,
            lexicalRetriever = lexicalRetriever,
            reranker = reranker,
            chunks = chunks,
            chunkVecs = chunkVecs
        )
        calibrationMs = calMs
        calibrationSkipped = calSkipped

        agroTool = AgroKnowledgeTool(
            ollama = ollama,
            retriever = retriever,
            lexicalRetriever = lexicalRetriever,
            reranker = reranker,
            chunks = chunks,
            chunkVecs = chunkVecs,
            relevanceThreshold = threshold
        )
    } else {
        println(
            "[1-4/4] вопрос не требует локального документа " +
                    "(needsDocument=false) — пропускаю PDF/эмбеддинги/калибровку."
        )
    }

    val ollamaKoog = buildOllamaKoogClient()
    val webSearchTool = buildWebSearchTool(client)

    val agent = buildAgent(
        ollamaKoog = ollamaKoog,
        agroTool = agroTool,
        webSearchTool = webSearchTool
    )

    lateinit var evalRun: AgentRunResult
    var groundedness: GroundednessJudgeResult? = null

    val agentMs = timed {
        evalRun = runAgentForEval(
            agent = agent,
            agroTool = agroTool,
            webSearchTool = webSearchTool,
            question = question
        )
    }.second

    /*
     * groundedness judge — отдельная LLM-проверка ПОСЛЕ готового ответа
     * (не влияет на evalRun.answer), нужна для отладки RAG-пайплайна.
     * В готовом проекте это лишний LLM-вызов на каждый вопрос — можно
     * выключить через SKIP_GROUNDEDNESS=1, как и калибровку.
     */
    val skipGroundedness = System.getenv("SKIP_GROUNDEDNESS") == "1"
    val groundednessMs = timed {
        if (skipGroundedness) {
            println("groundedness judge пропущен (SKIP_GROUNDEDNESS=1)")
        } else {
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
    }.second

    printRunResult(evalRun, groundedness)
    printTiming(
        TimingReport(
            gateMs = gateMs,
            documentSkipped = !classification.needsDocument,
            embeddingsMs = embeddingsMs,
            calibrationMs = calibrationMs,
            calibrationSkipped = calibrationSkipped,
            agentMs = agentMs,
            groundednessMs = groundednessMs,
            groundednessSkipped = skipGroundedness
        )
    )

    agent.close()
    reranker.close()
    ollama.close()
}

private fun promptForPdfPath(): String {
    print("Путь к PDF-файлу: ")
    return readlnOrNull().orEmpty()
        .trim()
        .trim('"')
}

private fun promptForQuestion(): String {
    print("Твой вопрос к документу: ")
    return readlnOrNull().orEmpty()
        .trim()
}

private fun loadDocumentChunks(pdfPath: String): List<DocumentChunk> {
    println("[1/4] читаю PDF...")
    val pages = readPdfPages(pdfPath)
    println("      страниц: ${pages.size}")

    println("[2/4] режу на чанки...")
    val chunks = splitIntoChunks(pages)
    println("      чанков: ${chunks.size}")

    return chunks
}

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
private suspend fun resolveRelevanceThreshold(
    ollama: OllamaClient,
    retriever: Retriever,
    lexicalRetriever: LexicalRetriever,
    reranker: BgeReranker,
    chunks: List<DocumentChunk>,
    chunkVecs: List<List<Double>>
): Triple<Float, Long, Boolean> {

    val skipCalibration = System.getenv("SKIP_CALIBRATION") == "1"

    var threshold = RELEVANCE_THRESHOLD
    val calibrationMs = timed {
        if (skipCalibration) {
            println(
                "[4/4] калибровка пропущена (SKIP_CALIBRATION=1), " +
                        "используется RELEVANCE_THRESHOLD=$RELEVANCE_THRESHOLD"
            )
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
    }.second

    return Triple(threshold, calibrationMs, skipCalibration)
}

private fun buildOllamaKoogClient(): OllamaClientKoog {
    @Suppress("UnstableApiUsage")
    val defaultFactory = HttpClientFactoryResolver.resolve()
    val loggingFactory = LoggingKoogHttpClientFactory(defaultFactory)

    @Suppress("UnstableApiUsage")
    return OllamaClientKoog(
        httpClientFactory = loggingFactory,
        /*
         * Без явного num_ctx Ollama сама выбирает размер контекстного
         * окна (см. AppConfig.OLLAMA_NUM_CTX) — на этой машине это
         * заставляло ~72% слоёв модели уходить на CPU, потому что
         * KV-cache под такой контекст не помещался в VRAM видеокарты.
         */
        contextWindowStrategy = ContextWindowStrategy.Companion.Fixed(OLLAMA_NUM_CTX)
    )
}

/*
 * Web search сейчас опционален: без переменной окружения (или
 * при недоступном API) агент просто работает с одним searchKnowledge,
 * вместо того чтобы падать после уже проделанной дорогой работы
 * (PDF -> чанки -> эмбеддинги -> калибровка). Реальные сбои самого
 * запроса к API (сеть, лимиты) ловятся внутри WebSearchTool.execute
 * и тоже не роняют агента.
 *
 * Провайдер — Serper.dev, не Yandex Search: Yandex недоступен из
 * Украины (верификация оплаты отклоняет украинские номера), см.
 * minirag.web.YandexSearchClient.
 */
private fun buildWebSearchTool(client: HttpClient): WebSearchTool? {
    val apiKey = System.getenv(SERPER_API_KEY_ENV)

    if (apiKey.isNullOrBlank()) {
        println(
            "[webSearch] $SERPER_API_KEY_ENV не задан — webSearch отключён, " +
                    "доступен только searchKnowledge."
        )
        return null
    }

    val gl = System.getenv(SERPER_GL_ENV) ?: SERPER_DEFAULT_GL
    val hl = System.getenv(SERPER_HL_ENV) ?: SERPER_DEFAULT_HL

    val webSearchClient: WebSearchClient = SerperSearchClient(
        httpClient = client,
        apiKey = apiKey,
        gl = gl,
        hl = hl
    )

    return WebSearchTool(searchClient = webSearchClient)
}

private fun buildAgent(
    ollamaKoog: OllamaClientKoog,
    agroTool: AgroKnowledgeTool?,
    webSearchTool: WebSearchTool?
): AIAgent<String, String> {

    val tools: List<Tool<*, *>> = listOfNotNull(agroTool, webSearchTool)

    return AIAgent(
        promptExecutor = MultiLLMPromptExecutor(ollamaKoog),
        llmModel = gemma4,
        temperature = 0.0,
        systemPrompt = buildSystemPrompt(
            agroAvailable = agroTool != null,
            webSearchAvailable = webSearchTool != null
        ),
        toolRegistry = ToolRegistry { tools(tools) },
        strategy = agroStrategy,
    )
}

private fun buildSystemPrompt(agroAvailable: Boolean, webSearchAvailable: Boolean): String {
    val agroDescription = """
        searchKnowledge:
        используй для информации из локальной агрономической базы знаний:
        - болезни;
        - симптомы;
        - возбудители;
        - условия развития;
        - меры борьбы;
        - лечение;
        - профилактика;
        - содержание документов.
    """.trimIndent()

    val webSearchDescription = """
        webSearch:
        используй для актуальной информации из интернета:
        - текущие цены;
        - наличие препаратов;
        - где купить препарат;
        - актуальные предложения продавцов;
        - текущую доступность препаратов.
    """.trimIndent()

    /*
     * Оба тула опциональны (needsDocument=false отсекает searchKnowledge
     * ещё до сборки агента, чтобы не гонять PDF/эмбеддинги зря; webSearch
     * недоступен без SERPER_API_KEY) — системный промпт собирается только
     * из реально доступных тулов, а не описывает несуществующий выбор.
     */
    return when {
        agroAvailable && webSearchAvailable -> """
У тебя есть два инструмента.

$agroDescription

$webSearchDescription

Если пользователь спрашивает содержание локального документа,
используй searchKnowledge и НЕ используй webSearch.

Если пользователь спрашивает актуальную цену, наличие
или место покупки, используй webSearch.

Если вопрос требует информации и из документа,
и из интернета, используй оба инструмента.

Не выдумывай цены, наличие или предложения,
если webSearch их не вернул.

Не выдавай информацию из webSearch как информацию,
содержащуюся в локальном документе.
""".trimIndent()

        agroAvailable -> "У тебя есть инструмент searchKnowledge.\n\n$agroDescription"

        webSearchAvailable -> """
У тебя есть инструмент webSearch.

$webSearchDescription

Не выдумывай цены, наличие или предложения,
если webSearch их не вернул.
""".trimIndent()

        else -> "У тебя нет инструментов — отвечай на основе собственных знаний."
    }
}

private fun printRunResult(
    evalRun: AgentRunResult,
    groundedness: GroundednessJudgeResult?
) {
    println()
    println("========== TOOL ==========")
    println("searchKnowledge called: ${evalRun.toolCalled}")
    println("webSearch called: ${evalRun.webSearchCalled}")

    println()
    println("========== RETRIEVED CONTEXT ==========")
    println(evalRun.retrievedContext ?: "<NO TOOL RESULT>")

    println()
    println("========== FINAL ANSWER ==========")
    println(evalRun.answer)

    println()
    println("========== GROUNDEDNESS ==========")
    if (groundedness == null) {
        println("пропущено (SKIP_GROUNDEDNESS=1)")
    } else {
        println("grounded: ${groundedness.grounded}")
        groundedness.unsupportedClaims.forEach {
            println("- $it")
        }
    }

    println("==================================")
}

private fun printTiming(report: TimingReport) {
    println()
    println("========== TIMING ==========")
    println("domain gate: ${report.gateMs}мс")
    if (report.documentSkipped) {
        println("эмбеддинги: пропущены (needsDocument=false)")
        println("калибровка: пропущена (needsDocument=false)")
    } else {
        println("эмбеддинги: ${report.embeddingsMs}мс")
        println(
            if (report.calibrationSkipped) "калибровка: пропущена"
            else "калибровка: ${report.calibrationMs}мс"
        )
    }
    println("  agent (tool-call + final answer): ${report.agentMs}мс")
    println(
        if (report.groundednessSkipped) "  groundedness judge: пропущен"
        else "  groundedness judge: ${report.groundednessMs}мс"
    )
    println("ответ на вопрос: ${report.answerMs}мс")
    println("=============================")
}

private suspend inline fun <T> timed(block: suspend () -> T): Pair<T, Long> {
    val start = System.currentTimeMillis()
    val result = block()
    return result to (System.currentTimeMillis() - start)
}
