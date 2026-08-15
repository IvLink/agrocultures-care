package minirag

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.http.client.HttpClientFactoryResolver
import ai.koog.prompt.executor.llms.MultiLLMPromptExecutor
import ai.koog.prompt.llm.LLMCapability
import ai.koog.prompt.llm.LLMProvider
import ai.koog.prompt.llm.LLModel
import minirag.agents.AgroKnowledgeTool
import minirag.agents.TestTool
import ai.koog.prompt.executor.ollama.client.OllamaClient as OllamaClientKoog
import minirag.bge_reranker.BgeReranker
import minirag.config.createHttpClient
import minirag.network.LoggingKoogHttpClientFactory
import minirag.ollama.OllamaClient
import minirag.pdf.readPdf
import minirag.retrieval.Retriever
import minirag.strategy.agroStrategy
import minirag.text.splitIntoChunks

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

//
//    ollama.testDirectToolResult()
//
//    ollama.close()
//    client.close()
//    return

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
    val text = readPdf(pdfPath)
    println("      символов: ${text.length}")

    println("[2/4] режу на чанки...")
    val chunks = splitIntoChunks(text)
    println("      чанков: ${chunks.size}")

    chunks.forEachIndexed { index, chunk ->
        if (
            chunk.contains("Фитофтороз", ignoreCase = true) ||
            chunk.contains("Phytophthora", ignoreCase = true)
        ) {
            println("\n========== CHUNK $index ==========")
            println(chunk)
            println("========== END CHUNK $index ==========")
        }
    }

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
        Ты агро-помощник.

        Если для ответа нужна информация из локальной базы знаний,
        используй инструмент searchKnowledge.

        Не придумывай сведения, которых нет в результате поиска.
    """.trimIndent(),
        toolRegistry = ToolRegistry {
            tools(agroTool)
        },
        strategy = agroStrategy
    )

//    println("[4/4] анализирую вопрос...")
//
//    val queries = ollama.decomposeQuestion(question)
//
//    println("\n[queries]")
//    queries.forEach {
//        println(" - $it")
//    }
//    println("\n[поиск]")
//
//    val candidateChunks = retriever.multiQueryRetrieval(
//        questions = queries,
//        chunks = chunks,
//        chunkVecs = chunkVecs
//    ).distinct()
//
//    println("\n[reranker] кандидатов: ${candidateChunks.size}")
//
//    val rerankResults = reranker.multiQueryRerank(
//        queries = queries,
//        chunks = candidateChunks,
//        topN = TOP_RERANK
//    )
//
//    println(
//        "[reranker] оставлено: ${
//            rerankResults.sumOf { it.chunks.size }
//        } фрагментов до distinct"
//    )

//    val result = agent.run(
//        "Используй testTool и скажи, какое число он вернул."
//    )
//
//    println("\n========== FINAL ANSWER ==========")
//    println(result)


    println("\n================ ОТВЕТ ================\n")
    val result = agent.run(question)
    println(result)


//    println(
//        ollama.answer(
//            question = question,
//            rerankResults = rerankResults
//        )
//    )

    agent.close()
    reranker.close()
    ollama.close()
}
