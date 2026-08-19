package minirag.bge_reranker

import ai.djl.huggingface.tokenizers.Encoding
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import ai.djl.util.PairList
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

/*
 * Сколько (query, window) пар отправляется в ONNX за один
 * session.run(). Раньше на каждое окно каждого кандидата уходил
 * ОТДЕЛЬНЫЙ inference — а именно это, а не логика самого reranker'а,
 * было основным источником задержки: десятки последовательных
 * ONNX forward pass на один вопрос пользователя. Батчинг не меняет
 * логику ранжирования (то же окно, тот же score), только объединяет
 * несколько независимых inference в один вызов. Ограничение размера
 * батча — просто чтобы не строить один гигантский тензор, если
 * candidate pool когда-нибудь станет намного больше.
 */
private const val ONNX_BATCH_SIZE = 32

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
     * длину chunk — а значит collectWindows() никогда бы
     * не заходил в ветку с окнами (см. ниже).
     *
     * padding не трогаем — оставляем дефолт (LONGEST): для
     * batchEncode() ниже это как раз то, что нужно, чтобы весь
     * батч пришёл уже выровненным по длине самого длинного
     * элемента батча, без ручного паддинга.
     */
    private val tokenizer =
        HuggingFaceTokenizer.builder()
            .optTokenizerPath(Paths.get(tokenizerPath))
            .optTruncation(false)
            .build()

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

    /*
     * Считает logits для целого батча (query, window) пар ОДНИМ
     * ONNX forward pass'ом, а не по одному на пару.
     *
     * Требует, чтобы все encodings в батче были одной длины —
     * это гарантируется тем, что они получены из ОДНОГО вызова
     * tokenizer.batchEncode() (см. rerank ниже), а не собраны
     * из разных вызовов encode().
     */
    private fun scoreBatch(
        encodings: List<Encoding>
    ): FloatArray {

        if (encodings.isEmpty()) {
            return FloatArray(0)
        }

        val inputIdsTensor =
            OnnxTensor.createTensor(
                environment,
                Array(encodings.size) { encodings[it].ids }
            )

        val attentionMaskTensor =
            OnnxTensor.createTensor(
                environment,
                Array(encodings.size) { encodings[it].attentionMask }
            )

        val inputs = mapOf(
            "input_ids" to inputIdsTensor,
            "attention_mask" to attentionMaskTensor
        )

        try {

            session.run(inputs).use { result ->

                return when (
                    val logits = result[0].value
                ) {

                    is Array<*> -> FloatArray(encodings.size) { i ->
                        (logits[i] as FloatArray)[0]
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

    /**
     * Разбивает document на окна ≤512 токенов (сам текст, без
     * инференса — токенизация нужна только чтобы проверить длину,
     * а не для scoring, поэтому остаётся дешёвой, не батчится).
     *
     * Если document помещается в 512 токенов целиком —
     * одно окно, равное всему document.
     *
     * Иначе:
     *
     * document
     *   ↓
     * windows по символам
     *   ↓
     * каждое окно ужимается, пока не влезет в RERANK_MAX_TOKENS
     */
    private fun collectWindows(
        query: String,
        document: String
    ): List<String> {

        val fullEncoding = tokenizer.encode(query, document)
        println("full encoded tokens: " + fullEncoding.ids.size)

        if (fullEncoding.ids.size <= RERANK_MAX_TOKENS) {
            return listOf(document)
        }

        val windowSizeChars = maxOf(CHUNK_SIZE / 2, MIN_RERANK_WINDOW_SIZE)
        val overlapChars = maxOf(OVERLAP / 2, MIN_RERANK_OVERLAP)
        var start = 0
        val windows = mutableListOf<String>()

        while (start < document.length) {
            val end = minOf(start + windowSizeChars, document.length)
            var window = document
                .substring(start, end)
                .trim()
            var encoding = tokenizer.encode(query, window)

            while (encoding.ids.size > RERANK_MAX_TOKENS) {
                val newLength = maxOf(1, (window.length * 0.8).toInt())
                window = window.substring(0, newLength)
                    .trim()
                encoding = tokenizer.encode(query, window)
            }

            windows += window

            if (end == document.length) {
                break
            }
            start = maxOf(start + 1, end - overlapChars)
        }

        return windows
    }

    /**
     * Оценивает весь candidate pool одним batched inference (плюс
     * ещё по одному на каждые ONNX_BATCH_SIZE окон, если окон
     * больше) и решает, какие chunks действительно релевантны.
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

        val validCandidateIds = candidates.mapNotNull { candidate ->
            if (chunksById.containsKey(candidate.chunkId)) {
                candidate.chunkId
            } else {
                null
            }
        }

        /*
         * candidateIndex ссылается на позицию в validCandidateIds —
         * так каждое окно знает, к какому кандидату вернуть свой score.
         */
        data class WindowJob(val candidateIndex: Int, val text: String)

        val jobs = mutableListOf<WindowJob>()

        validCandidateIds.forEachIndexed { candidateIndex, chunkId ->
            val document = chunksById.getValue(chunkId).retrievalText()
            collectWindows(query, document).forEach { window ->
                jobs += WindowJob(candidateIndex, window)
            }
        }

        val bestRawScore = FloatArray(validCandidateIds.size) {
            Float.NEGATIVE_INFINITY
        }

        jobs.chunked(ONNX_BATCH_SIZE).forEach { batch ->

            val pairs = PairList(
                batch.map { query },
                batch.map { it.text }
            )

            val encodings = tokenizer.batchEncode(pairs)
            val rawScores = scoreBatch(encodings.toList())

            batch.forEachIndexed { i, job ->
                val rawScore = rawScores[i]

                println(
                    "window(candidate=${job.candidateIndex}) " +
                            "tokens=${encodings[i].ids.size} " +
                            "raw=${"%.4f".format(rawScore)} " +
                            "normalized=${"%.4f".format(sigmoid(rawScore))}"
                )

                if (rawScore > bestRawScore[job.candidateIndex]) {
                    bestRawScore[job.candidateIndex] = rawScore
                }
            }
        }

        val results = validCandidateIds
            .mapIndexed { candidateIndex, chunkId ->
                val rawScore = bestRawScore[candidateIndex]
                val normalizedScore = sigmoid(rawScore)

                RerankResult(
                    chunkId = chunkId,
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
