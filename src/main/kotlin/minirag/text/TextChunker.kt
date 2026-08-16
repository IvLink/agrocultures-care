package minirag.text

import minirag.config.CHUNK_SIZE
import minirag.config.OVERLAP

/**
 * Универсальный chunker документов.
 *
 * Основной принцип:
 *
 * 1. Сначала сохраняем естественные границы страниц.
 * 2. Если страница помещается в CHUNK_SIZE,
 *    она остаётся одним chunk.
 * 3. Если страница большая, режем её по абзацам.
 * 4. Если абзац слишком большой, режем его с overlap.
 *
 * Chunker НЕ знает:
 * - что такое болезнь;
 * - что такое возбудитель;
 * - где заголовок;
 * - какой у документа тип;
 * - какие поля есть внутри документа.
 *
 * Поэтому один и тот же код работает для:
 * - справочников;
 * - инструкций;
 * - учебников;
 * - технической документации;
 * - документов по удобрениям;
 * - документов по профилактике;
 * - и т.д.
 */
fun splitIntoChunks(text: String): List<String> {
    val normalized = normalizeText(text)

    if (normalized.isBlank()) {
        return emptyList()
    }

    /*
     * PDFTextStripper обычно сохраняет границу страницы
     * через form-feed (\u000C).
     *
     * Если документ пришёл не из PDF и такой границы нет,
     * весь документ рассматривается как один поток текста.
     */
    val pages = normalized
        .split('\u000C')
        .map { it.trim() }
        .filter { it.isNotBlank() }

    return pages.flatMap { page ->
        chunkPage(page)
    }
}

/**
 * Нормализуем технический мусор, но НЕ уничтожаем структуру документа.
 */
private fun normalizeText(text: String): String {
    return text
        .replace("\r\n", "\n")
        .replace('\r', '\n')
        /*
         * Не трогаем \u000C.
         * Это граница страницы, она нужна выше.
         */
        .replace(Regex("[ \t]+"), " ")
        .replace(Regex("\n{3,}"), "\n\n")
        .trim()
}

/**
 * Обрабатывает одну страницу.
 */
private fun chunkPage(page: String): List<String> {

    /*
     * Если вся страница помещается в chunk,
     * НЕ РЕЖЕМ ЕЁ.
     *
     * Это ключевой момент.
     *
     * Например, если на странице находятся:
     *
     * Название
     * Возбудитель
     * Симптомы
     * Условия
     * Меры борьбы
     *
     * всё это остаётся одним embedding.
     */
    if (page.length <= CHUNK_SIZE) {
        return listOf(page)
    }

    /*
     * Большую страницу сначала пытаемся разбить
     * по естественным абзацам.
     */
    val paragraphs = page
        .split(Regex("""\n\s*\n+"""))
        .map { it.trim() }
        .filter { it.isNotBlank() }

    /*
     * Если PDF/text extractor не оставил пустых строк,
     * разбиваем по строкам.
     */
    if (paragraphs.size <= 1) {
        return chunkLongText(page)
    }

    val chunks = mutableListOf<String>()

    var current = mutableListOf<String>()
    var currentLength = 0

    fun flush() {
        if (current.isEmpty()) {
            return
        }

        val chunk = current
            .joinToString("\n\n")
            .trim()

        if (chunk.isNotBlank()) {
            chunks += chunk
        }

        current = mutableListOf()
        currentLength = 0
    }

    for (paragraph in paragraphs) {

        /*
         * Если сам абзац больше CHUNK_SIZE,
         * сначала закрываем текущий chunk.
         */
        if (paragraph.length > CHUNK_SIZE) {
            flush()

            /*
             * Затем режем большой абзац отдельно.
             */
            chunks += chunkLongText(paragraph)

            continue
        }

        /*
         * Абзац помещается в текущий chunk.
         */
        if (
            currentLength == 0 ||
            currentLength + paragraph.length + 2 <= CHUNK_SIZE
        ) {
            current += paragraph

            currentLength += if (currentLength == 0) {
                paragraph.length
            } else {
                paragraph.length + 2
            }

            continue
        }

        /*
         * Следующий абзац уже не помещается.
         */
        flush()

        current += paragraph
        currentLength = paragraph.length
    }

    flush()

    /*
     * После первичного разбиения добавляем overlap
     * между соседними chunks.
     *
     * Это особенно важно, когда смысл предложения
     * или секции находится на границе двух chunks.
     */
    return addOverlap(chunks)
}

/**
 * Разбивает очень большой текст.
 *
 * Сначала стараемся резать по строкам,
 * затем по предложениям,
 * и только если это невозможно,
 * режем по символам.
 */
private fun chunkLongText(text: String): List<String> {

    if (text.length <= CHUNK_SIZE) {
        return listOf(text.trim())
    }

    /*
     * Сначала используем строки.
     */
    val lines = text
        .split('\n')
        .map { it.trim() }
        .filter { it.isNotBlank() }

    if (lines.size > 1) {
        val chunks = packParts(lines)
        return addOverlap(chunks)
    }

    /*
     * Если всё оказалось одной строкой,
     * пробуем предложения.
     */
    val sentences = text
        .split(Regex("""(?<=[.!?。！？])\s+"""))
        .map { it.trim() }
        .filter { it.isNotBlank() }

    if (sentences.size > 1) {
        val chunks = packParts(sentences)
        return addOverlap(chunks)
    }

    /*
     * Совсем крайний случай:
     * непрерывная строка длиннее CHUNK_SIZE.
     */
    return splitByCharacters(text)
}

/**
 * Упаковывает последовательность частей в chunks.
 */
private fun packParts(parts: List<String>): List<String> {

    val chunks = mutableListOf<String>()

    var current = mutableListOf<String>()
    var currentLength = 0

    fun flush() {
        if (current.isEmpty()) {
            return
        }

        chunks += current
            .joinToString("\n\n")
            .trim()

        current = mutableListOf()
        currentLength = 0
    }

    for (part in parts) {

        if (part.length > CHUNK_SIZE) {
            flush()
            chunks += splitByCharacters(part)
            continue
        }

        val separatorLength =
            if (current.isEmpty()) 0 else 2

        if (
            currentLength +
            separatorLength +
            part.length <= CHUNK_SIZE
        ) {
            current += part
            currentLength += separatorLength + part.length
        } else {
            flush()

            current += part
            currentLength = part.length
        }
    }

    flush()

    return chunks
}

/**
 * Добавляет overlap между соседними chunks.
 *
 * Берём конец предыдущего chunk и добавляем его
 * в начало следующего.
 *
 * При этом не допускаем бесконечного роста chunk.
 */
private fun addOverlap(
    chunks: List<String>
): List<String> {

    if (chunks.size <= 1) {
        return chunks
    }

    val result = mutableListOf<String>()

    for (index in chunks.indices) {

        val current = chunks[index]

        if (index == 0) {
            result += current
            continue
        }

        val previous = chunks[index - 1]

        val overlapSize = minOf(
            OVERLAP,
            previous.length
        )

        val overlap = previous
            .takeLast(overlapSize)
            .trim()

        if (overlap.isBlank()) {
            result += current
            continue
        }

        /*
         * Если overlap + current слишком большой,
         * уменьшаем overlap.
         */
        val maxOverlap = maxOf(
            0,
            CHUNK_SIZE - current.length - 1
        )

        val actualOverlap = overlap
            .takeLast(minOf(overlap.length, maxOverlap))
            .trim()

        if (actualOverlap.isBlank()) {
            result += current
        } else {
            result += "$actualOverlap\n\n$current"
        }
    }

    return result
}

/**
 * Последний fallback для текста,
 * который невозможно естественно разделить.
 */
private fun splitByCharacters(
    text: String
): List<String> {

    val result = mutableListOf<String>()

    var start = 0

    while (start < text.length) {

        val end = minOf(
            start + CHUNK_SIZE,
            text.length
        )

        result += text
            .substring(start, end)
            .trim()

        if (end == text.length) {
            break
        }

        /*
         * Не допускаем отрицательного шага,
         * даже если кто-то поставит OVERLAP >= CHUNK_SIZE.
         */
        val step = maxOf(
            1,
            CHUNK_SIZE - OVERLAP
        )

        start += step
    }

    return result
}
