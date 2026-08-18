package minirag.text

import minirag.config.CHUNK_SIZE
import minirag.config.OVERLAP
import minirag.models.DocumentChunk
import minirag.models.DocumentPage

/**
 * Универсальный chunker документов.
 *
 * Chunker не знает предметную область документа: он не ищет
 * слова вроде "болезни" или "меры борьбы". Вместо этого он
 * распознаёт СТРУКТУРУ, которую сам документ уже задаёт —
 * короткие строки-заголовки вида "Заголовок:" — и сохраняет
 * эту структуру как section у каждого chunk.
 *
 * Chunker сохраняет:
 * - исходную страницу (реальную, из DocumentPage);
 * - раздел документа (section), если он определяется структурой;
 * - естественные границы текста;
 * - id chunk.
 */
fun splitIntoChunks(pages: List<DocumentPage>): List<DocumentChunk> {

    var nextChunkId = 0
    val result = mutableListOf<DocumentChunk>()

    for (page in pages) {

        val normalized = normalizeText(page.text)

        if (normalized.isBlank()) {
            continue
        }

        val pageBlocks = splitIntoSectionBlocks(normalized)

        for (block in pageBlocks.blocks) {

            chunkText(block.text)
                .forEach { text ->

                    result += DocumentChunk(
                        id = nextChunkId++,
                        page = page.number,
                        text = text,
                        section = block.section,
                        title = pageBlocks.strayTitle
                    )
                }
        }
    }

    return result
}

private fun normalizeText(text: String): String {
    return text
        .replace("\r\n", "\n")
        .replace('\r', '\n')
        .replace(Regex("[ \t]+"), " ")
        .replace(Regex("\n{3,}"), "\n\n")
        .trim()
}

private data class SectionBlock(
    val section: String?,
    val text: String
)

/*
 * Заголовок раздела в этих документах — это короткая
 * самостоятельная строка, заканчивающаяся двоеточием,
 * например:
 *
 *   Возбудитель болезни:
 *   Меры борьбы:
 *   Симптомы:
 *
 * Это не привязано к конкретным словам предметной области —
 * работает для любого документа со структурой "Заголовок:".
 */
private val sectionHeadingRegex =
    Regex("""^[А-ЯЁA-Z][А-Яа-яЁёA-Za-z0-9 \-/]{1,58}:$""")

private fun detectSectionHeading(line: String): String? {

    val trimmed = line.trim()

    if (!sectionHeadingRegex.matches(trimmed)) {
        return null
    }

    return trimmed.removeSuffix(":").trim()
}

private val sentenceTerminators = setOf('.', '!', '?')

/*
 * Многие PDF со сложной вёрсткой (текстовая колонка + колонка
 * подписей к фото/заголовок записи рядом) извлекаются
 * PDFTextStripper НЕ в визуальном порядке: заголовок записи
 * и подписи к иллюстрациям физически лежат в content stream
 * после всей текстовой колонки и "приезжают" в конец последнего
 * распознанного раздела (обычно последнего по странице), а не
 * туда, где они на самом деле находятся на странице.
 *
 * Единственный надёжный (и не привязанный к словам конкретного
 * документа) сигнал такой "уехавшей" строки — она СТРУКТУРНО
 * не похожа на продолжение прозы:
 *
 * - идёт сразу после уже законченного предложения (иначе это
 *   вторая строка того же предложения, а не новая мысль);
 * - сама короткая и самостоятельная, без завершающей пунктуации;
 * - и, что важнее всего, СЛЕДУЮЩАЯ строка не продолжает её —
 *   не начинается со строчной буквы. Перенос предложения на
 *   новую строку (обычное дело при постраничном извлечении PDF)
 *   почти всегда продолжается строчной буквой на следующей
 *   строке; самостоятельный заголовок — нет.
 */
private fun looksLikeStrayTitleLine(
    trimmed: String,
    nextNonBlankLine: String?
): Boolean {

    /*
     * Заголовки записей не обязаны быть многословными
     * ("Столбур" — такой же валидный заголовок, как и
     * "Фитофтороз (фитофторозная гниль) пасленовых") —
     * защиту от случайных коротких обрывков даёт не длина
     * или количество слов, а требование "после законченного
     * предложения" + "не продолжается следующей строкой".
     */
    if (trimmed.length !in 4..70) {
        return false
    }

    if (!trimmed.first().isUpperCase()) {
        return false
    }

    if (trimmed.last() in sentenceTerminators) {
        return false
    }

    if (nextNonBlankLine != null && nextNonBlankLine.first().isLowerCase()) {
        return false
    }

    return true
}

private data class PageBlocks(
    val blocks: List<SectionBlock>,
    val strayTitle: String?
)

private fun splitIntoSectionBlocks(
    pageText: String
): PageBlocks {

    val lines = pageText.split('\n')

    val blocks = mutableListOf<SectionBlock>()

    var currentSection: String? = null
    var buffer = StringBuilder()
    var bufferHasContent = false
    var previousLineEndedSentence = false
    var strayTitle: String? = null

    fun flush() {

        val text = buffer.toString().trim()

        if (text.isNotBlank()) {
            blocks += SectionBlock(
                section = currentSection,
                text = text
            )
        }

        buffer = StringBuilder()
        bufferHasContent = false
    }

    for (index in lines.indices) {

        val rawLine = lines[index]
        val trimmedLine = rawLine.trim()
        val heading = detectSectionHeading(rawLine)

        if (heading != null) {
            flush()
            currentSection = heading
            previousLineEndedSentence = false
            buffer.append(rawLine).append('\n')
            continue
        }

        /*
         * Строка похожа на заголовок записи, "уехавший" в хвост
         * текущего раздела: у нас уже есть тело раздела, оно
         * закончено законченным предложением, а эта строка сама
         * на предложение не похожа и не продолжается следующей
         * строкой. Отрезаем её вместе со всем, что после неё,
         * в отдельный блок — не даём ей "загрязнить" смысл
         * текущего раздела, и запоминаем как заголовок записи
         * для всех chunks этой страницы.
         */
        if (
            strayTitle == null &&
            bufferHasContent &&
            previousLineEndedSentence &&
            looksLikeStrayTitleLine(
                trimmed = trimmedLine,
                nextNonBlankLine = lines
                    .asSequence()
                    .drop(index + 1)
                    .map { it.trim() }
                    .firstOrNull { it.isNotEmpty() }
            )
        ) {
            flush()
            strayTitle = trimmedLine
            currentSection = null
        }

        buffer.append(rawLine).append('\n')

        if (trimmedLine.isNotEmpty()) {
            bufferHasContent = true
            previousLineEndedSentence = trimmedLine.last() in sentenceTerminators
        }
    }

    flush()

    if (blocks.isEmpty()) {
        return PageBlocks(
            blocks = listOf(
                SectionBlock(
                    section = null,
                    text = pageText
                )
            ),
            strayTitle = strayTitle
        )
    }

    return PageBlocks(blocks = blocks, strayTitle = strayTitle)
}

private fun chunkText(text: String): List<String> {

    if (text.length <= CHUNK_SIZE) {
        return listOf(text)
    }

    val paragraphs = text
        .split(Regex("""\n\s*\n+"""))
        .map { it.trim() }
        .filter { it.isNotBlank() }

    if (paragraphs.size <= 1) {
        return chunkLongText(text)
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

        if (paragraph.length > CHUNK_SIZE) {

            flush()

            chunks += chunkLongText(paragraph)

            continue
        }

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

        flush()

        current += paragraph
        currentLength = paragraph.length
    }

    flush()

    return addOverlap(chunks)
}

private fun chunkLongText(
    text: String
): List<String> {

    if (text.length <= CHUNK_SIZE) {
        return listOf(text.trim())
    }

    val lines = text
        .split('\n')
        .map { it.trim() }
        .filter { it.isNotBlank() }

    if (lines.size > 1) {

        val chunks = packParts(lines)

        return addOverlap(chunks)
    }

    val sentences = text
        .split(Regex("""(?<=[.!?。！？])\s+"""))
        .map { it.trim() }
        .filter { it.isNotBlank() }

    if (sentences.size > 1) {

        val chunks = packParts(sentences)

        return addOverlap(chunks)
    }

    return splitByCharacters(text)
}

private fun packParts(
    parts: List<String>
): List<String> {

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

            currentLength +=
                separatorLength + part.length

        } else {

            flush()

            current += part
            currentLength = part.length
        }
    }

    flush()

    return chunks
}

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

        val maxOverlap = maxOf(
            0,
            CHUNK_SIZE - current.length - 1
        )

        val actualOverlap = overlap
            .takeLast(
                minOf(
                    overlap.length,
                    maxOverlap
                )
            )
            .trim()

        if (actualOverlap.isBlank()) {

            result += current

        } else {

            result += "$actualOverlap\n\n$current"
        }
    }

    return result
}

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

        val step = maxOf(
            1,
            CHUNK_SIZE - OVERLAP
        )

        start += step
    }

    return result
}
