package minirag

import minirag.bge_reranker.BgeReranker
import minirag.config.TOP_RERANK
import minirag.config.createHttpClient
import minirag.ollama.OllamaClient
import minirag.pdf.readPdf
import minirag.retrieval.Retriever
import minirag.text.splitIntoChunks

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
    println("[4/4] анализирую вопрос...")

    val queries = ollama.decomposeQuestion(question)

    println("\n[queries]")
    queries.forEach {
        println(" - $it")
    }
    println("\n[поиск]")

    val candidateChunks = retriever.multiQueryRetrieval(
        questions = queries,
        chunks = chunks,
        chunkVecs = chunkVecs
    ).distinct()

    println("\n[reranker] кандидатов: ${candidateChunks.size}")

    val rerankResults = reranker.multiQueryRerank(
        queries = queries,
        chunks = candidateChunks,
        topN = TOP_RERANK
    )

    println(
        "[reranker] оставлено: ${
            rerankResults.sumOf { it.chunks.size }
        } фрагментов до distinct"
    )
    println("\n================ ОТВЕТ ================\n")
    println(
        ollama.answer(
            question = question,
            rerankResults = rerankResults
        )
    )

    reranker.close()
    ollama.close()
}
