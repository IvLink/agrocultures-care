package minirag.text

import minirag.config.CHUNK_SIZE
import minirag.config.OVERLAP

// ── ШАГ 2: раздели text на чанки с перекрытием ─────────────────────
fun splitIntoChunks(text: String): List<String> {
    val paragraphs = text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .split(Regex("""\n\s*\n+"""))
        .map { it.trim() }
        .filter { it.length > 50 }

    val chunks = mutableListOf<String>()
    var current = mutableListOf<String>()
    var currentLength = 0

    fun flush() {
        if (current.isNotEmpty()) {
            chunks += current.joinToString("\n\n").trim()
            current = mutableListOf()
            currentLength = 0
        }
    }

    for (paragraph in paragraphs) {
        // Нормальный абзац помещается в текущий chunk.
        if (currentLength + paragraph.length <= CHUNK_SIZE) {
            current += paragraph
            currentLength += paragraph.length
            continue
        }

        // Закрываем текущий chunk.
        flush()

        // Сам абзац помещается в один chunk.
        if (paragraph.length <= CHUNK_SIZE) {
            current += paragraph
            currentLength = paragraph.length
            continue
        }

        // Абзац слишком большой.
        // Пробуем разбить его по предложениям.
        val sentences = paragraph
            .split(Regex("""(?<=[.!?])\s+"""))
            .map { it.trim() }
            .filter { it.isNotEmpty() }

        for (sentence in sentences) {
            // Даже отдельное предложение слишком большое.
            if (sentence.length > CHUNK_SIZE) {
                flush()
                var start = 0
                while (start < sentence.length) {
                    val end = minOf(
                        start + CHUNK_SIZE,
                        sentence.length
                    )
                    chunks += sentence
                        .substring(start, end)
                        .trim()
                    start += CHUNK_SIZE - OVERLAP
                }
                continue
            }

            if (currentLength + sentence.length <= CHUNK_SIZE) {
                current += sentence
                currentLength += sentence.length
            } else {
                flush()
                current += sentence
                currentLength = sentence.length
            }
        }
    }
    flush()
    return chunks
}
