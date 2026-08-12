package minirag.bge_reranker

import ai.djl.huggingface.tokenizers.Encoding
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import minirag.config.CHUNK_SIZE
import minirag.config.MIN_RERANK_OVERLAP
import minirag.config.MIN_RERANK_WINDOW_SIZE
import minirag.config.OVERLAP
import minirag.config.WINDOW_SIZE
import minirag.models.RerankResult
import java.nio.file.Paths

class BgeReranker(
    modelPath: String,
    tokenizerPath: String
) : AutoCloseable {

    private val environment = OrtEnvironment.getEnvironment()

    private val session = environment.createSession(
        modelPath,
        OrtSession.SessionOptions()
    )

    private val tokenizer = HuggingFaceTokenizer.builder()
        .optTokenizerPath(Paths.get(tokenizerPath))
        .optMaxLength(WINDOW_SIZE)
        .optTruncation(true)
        .build()

    private fun score(
        encoding: Encoding
    ): Float {
        val inputIds = encoding.ids
        val attentionMask = encoding.attentionMask
        println("tokens: ${inputIds.size}")

        val inputIdsTensor = OnnxTensor.createTensor(
            environment,
            arrayOf(inputIds)
        )
        val attentionMaskTensor = OnnxTensor.createTensor(
            environment,
            arrayOf(attentionMask)
        )
        val inputs = mapOf(
            "input_ids" to inputIdsTensor,
            "attention_mask" to attentionMaskTensor
        )

        try {
            session.run(inputs).use { result ->
                return when (val logits = result[0].value) {
                    is Array<*> -> {
                        val row = logits[0] as FloatArray
                        row[0]
                    }
                    else -> error(
                        "Unexpected logits type: ${logits::class.java}"
                    )
                }
            }
        } finally {
            inputIdsTensor.close()
            attentionMaskTensor.close()
        }
    }

    fun scoreChunk(
        query: String,
        document: String
    ): Float {
        val maxTokens = 512
        val fullEncoding = tokenizer.encode(query, document)

        if (fullEncoding.ids.size <= maxTokens) {
            println("full chunk: ${fullEncoding.ids.size} tokens")
            return score(fullEncoding)
        }
        println("chunk too large: ${fullEncoding.ids.size} tokens")

        val windowSizeChars = maxOf(CHUNK_SIZE / 2, MIN_RERANK_WINDOW_SIZE)
        val overlapChars = maxOf(OVERLAP / 2, MIN_RERANK_OVERLAP)
        var start = 0
        var windowIndex = 0
        var bestScore = Float.NEGATIVE_INFINITY

        while (start < document.length) {
            val end = minOf(start + windowSizeChars, document.length)
            var window = document
                .substring(start, end)
                .trim()

            // чтобы query + document помещались в 512 токенов.
            var encoding = tokenizer.encode(query, window)

            while (encoding.ids.size > maxTokens) {
                window = window
                    .substring(0, (window.length * 0.8).toInt())
                    .trim()

                if (window.isEmpty()) {
                    break
                }
                // Получаем новый Encoding после изменения окна.
                encoding = tokenizer.encode(query, window)
            }

            val tokenCount = encoding.ids.size
            val score = score(encoding)
            println(
                "window[$windowIndex] " +
                        "chars=${window.length} " +
                        "tokens=$tokenCount " +
                        "score=${"%.4f".format(score)}"
            )

            bestScore = maxOf(bestScore, score)
            windowIndex++
            if (end == document.length) {
                break
            }
            start = end - overlapChars
        }
        return bestScore
    }

    fun multiQueryRerank(
        queries: List<String>,
        chunks: List<String>,
        topN: Int
    ): List<RerankResult> {
        return queries.map { query ->
            println("\n[reranker] query: $query")
            val ranked = chunks
                .mapIndexed { index, chunk ->
                    val score = scoreChunk(
                        query = query,
                        document = chunk
                    )
                    index to score
                }
                .sortedByDescending { it.second }

            println("[reranker] оценки:")
            ranked.forEach { (index, score) ->
                println(
                    "chunk[$index] = ${"%.4f".format(score)}"
                )
            }

            val selected = ranked.take(topN)

            println(
                "[reranker] TOP-$topN: ${
                    selected.map { it.first }
                }"
            )
            RerankResult(
                query = query,
                chunks = selected.map { chunks[it.first] }
            )
        }
    }

    override fun close() {
        tokenizer.close()
        session.close()
    }
}