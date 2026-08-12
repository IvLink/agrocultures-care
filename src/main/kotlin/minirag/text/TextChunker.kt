package minirag.text

import minirag.config.CHUNK_SIZE
import minirag.config.OVERLAP

// ── ШАГ 2: раздели text на чанки с перекрытием ─────────────────────
fun splitIntoChunks(text: String): List<String> {
    return generateSequence(0) { it + (CHUNK_SIZE - OVERLAP) }
        .takeWhile { it < text.length }
        .map { start ->
            text
                .substring(start, (start + CHUNK_SIZE).coerceAtMost(text.length))
                .trim()
        }
        .filter { it.length > 50 }
        .toList()
}
