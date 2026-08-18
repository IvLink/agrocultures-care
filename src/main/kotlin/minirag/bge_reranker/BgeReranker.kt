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
import minirag.config.RERANK_MAX_TOKENS
import minirag.models.DocumentChunk
import minirag.models.RerankResult
import minirag.models.RetrievedChunk
import java.nio.file.Paths
import kotlin.math.exp

class BgeReranker(
    modelPath: String,
    tokenizerPath: String
) : AutoCloseable {

    private val environment = OrtEnvironment.getEnvironment()

    private val session =
        environment.createSession(
            modelPath,
            OrtSession.SessionOptions()
        )

    /*
     * ВАЖНО:
     *
     * truncation = false, иначе DJL молча обрезает
     * encode() до modelMaxLength (512 у этой модели)
     * ещё ДО того, как мы успеваем увидеть реальную
     * длину chunk — а значит scoreChunk() никогда бы
     * не заходил в ветку с окнами (см. ниже).
     */
    private val tokenizer =
        HuggingFaceTokenizer.builder()
            .optTokenizerPath(Paths.get(tokenizerPath))
            .optTruncation(false)
            .build()

    private fun score(
        encoding: Encoding
    ): Float {

        val inputIds = encoding.ids
        val attentionMask = encoding.attentionMask

        println(
            "tokens: ${inputIds.size}"
        )

        val inputIdsTensor =
            OnnxTensor.createTensor(
                environment,
                arrayOf(inputIds)
            )

        val attentionMaskTensor =
            OnnxTensor.createTensor(
                environment,
                arrayOf(attentionMask)
            )

        val inputs = mapOf(
            "input_ids" to inputIdsTensor,
            "attention_mask" to attentionMaskTensor
        )

        try {

            session.run(inputs)
                .use { result ->

                    return when (
                        val logits = result[0].value
                    ) {

                        is Array<*> -> {

                            val row =
                                logits[0] as FloatArray

                            row[0]
                        }

                        else -> error(
                            "Unexpected logits type: " +
                                    logits::class.java
                        )
                    }
                }

        } finally {

            inputIdsTensor.close()
            attentionMaskTensor.close()
        }
    }

    private fun sigmoid(
        value: Float
    ): Float {

        return when {
            value >= 0f -> {
                (1.0 / (1.0 + exp(-value.toDouble()))).toFloat()
            }

            else -> {
                val e = exp(value.toDouble())
                (e / (1.0 + e)).toFloat()
            }
        }
    }

    /**
     * Оценивает весь документ.
     *
     * Если document помещается в 512 токенов,
     * оцениваем его целиком.
     *
     * Если нет:
     *
     * document
     *   ↓
     * windows <= 512 tokens
     *   ↓
     * score каждой window
     *   ↓
     * MAX
     *
     * MAX здесь означает:
     * если ответ на вопрос находится в конце
     * документа, reranker всё равно его увидит.
     */
    fun scoreChunk(
        query: String,
        document: String
    ): Float {

        val fullEncoding = tokenizer.encode(query, document)
        println("full encoded tokens: " + fullEncoding.ids.size)

        if (fullEncoding.ids.size <= RERANK_MAX_TOKENS) {
            val rawScore = score(fullEncoding)
            println(
                "single window raw=${
                    "%.4f".format(rawScore)
                } normalized=${
                    "%.4f".format(sigmoid(rawScore))
                }"
            )
            return rawScore
        }

        /*
         * Документ длиннее лимита.
         *
         * Используем окна по символам,
         * затем гарантируем, что каждое окно
         * реально помещается в 512 токенов.
         */
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
            var encoding = tokenizer.encode(query, window)

            /*
             * Пока window не помещается в
             * реальный лимит reranker.
             */
            while (encoding.ids.size > RERANK_MAX_TOKENS) {
                val newLength = maxOf(1, (window.length * 0.8).toInt())
                window = window.substring(0, newLength)
                    .trim()
                encoding = tokenizer.encode(query, window)
            }
            val rawScore = score(encoding)
            val normalizedScore = sigmoid(rawScore)

            println(
                "window[$windowIndex] " +
                        "chars=${window.length} " +
                        "tokens=${encoding.ids.size} " +
                        "raw=${"%.4f".format(rawScore)} " +
                        "normalized=${"%.4f".format(normalizedScore)}"
            )
            bestScore = maxOf(bestScore, rawScore)
            windowIndex++

            if (end == document.length) {
                break
            }
            start = maxOf(start + 1, end - overlapChars)
        }

        return bestScore
    }

    /**
     * Оценивает весь candidate pool одним запросом и решает,
     * какие chunks действительно релевантны.
     *
     * accepted определяется через threshold, откалиброванный
     * по eval-набору (см. minirag.eval.RerankerCalibration),
     * а не через разрыв до лучшего результата — margin относительно
     * первого места не говорит нам, релевантен ли вообще хоть один
     * chunk (для нерелевантного запроса "лучший" результат всё
     * равно существует, просто он тоже мусор).
     */
    fun rerank(
        query: String,
        candidates: List<RetrievedChunk>,
        chunks: List<DocumentChunk>,
        threshold: Float
    ): List<RerankResult> {
        val chunksById = chunks.associateBy { it.id }
        val results = candidates
            .mapNotNull { candidate ->
                val chunk = chunksById[candidate.chunkId] ?: return@mapNotNull null
                val rawScore = scoreChunk(query = query, document = chunk.retrievalText())
                val normalizedScore = sigmoid(rawScore)
                RerankResult(
                    chunkId = chunk.id,
                    rawScore = rawScore,
                    normalizedScore = normalizedScore,
                    accepted = normalizedScore >= threshold
                )
            }
            .sortedByDescending { it.rawScore }
        println()
        println("[reranker] query: $query")
        println("[reranker] threshold (normalized) = %.4f".format(threshold))

        results.forEach { result ->
            val chunk = chunksById[result.chunkId]
            println(
                "chunk[${result.chunkId}] " +
                        "page=${chunk?.page} " +
                        "section=${chunk?.section} " +
                        "raw=${"%.4f".format(result.rawScore)} " +
                        "normalized=${"%.4f".format(result.normalizedScore)} " +
                        if (result.accepted) "ACCEPT" else "REJECT"
            )
        }

        return results
    }

    override fun close() {
        tokenizer.close()
        session.close()
    }
}