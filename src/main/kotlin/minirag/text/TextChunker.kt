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

    val normalizedPages = pages
        .map { it.number to normalizeText(it.text) }
        .filter { it.second.isNotBlank() }

    if (normalizedPages.isEmpty()) {
        return emptyList()
    }

    /*
     * Раздел, с которого в этом документе ЭМПИРИЧЕСКИ начинается
     * любая запись — выводится из самого документа (первый раздел
     * первой встреченной последовательности), а не хардкодится
     * ("Возбудитель болезни" в частности) — см. discoverFirstSection
     * и её использование в resolveZoneOnStrayTitle.
     */
    val firstSectionMarker = discoverFirstSection(
        normalizedPages.map { it.second }
    )

    val blocks = splitIntoSectionBlocks(normalizedPages, firstSectionMarker)

    var nextChunkId = 0
    val result = mutableListOf<DocumentChunk>()

    for (block in blocks) {
        chunkText(block.text)
            .forEach { text ->

                result += DocumentChunk(
                    id = nextChunkId++,
                    page = block.page,
                    text = text,
                    section = block.section,
                    title = block.title
                )
            }
    }

    return result
}

/*
 * Порядок разделов внутри записи в этих документах ПОСТОЯНЕН
 * (например, всегда "Возбудитель болезни" -> "Распространение" ->
 * "Симптомы" -> "Условия развития болезни" -> "Меры борьбы"), но
 * какие именно это разделы и в каком порядке — знание конкретного
 * документа, а не универсальная константа. Поэтому раздел, которым
 * запись начинается, выводится из самого документа: это первый
 * заголовок, который вообще встречается при последовательном чтении
 * всех страниц — используется дальше как сигнал "эта последовательность
 * блоков начинается с начала записи, а не с середины" (см.
 * resolveZoneOnStrayTitle).
 */
private fun discoverFirstSection(pageTexts: List<String>): String? {
    for (text in pageTexts) {
        for (line in text.split('\n')) {
            detectSectionHeading(line)?.let { return it }
        }
    }
    return null
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
    val page: Int,
    val section: String?,
    val text: String,
    val title: String?
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

    /*
     * Смотрим на регистр первой БУКВЫ следующей строки, а не первого
     * символа буквально — перенос предложения может продолжиться
     * после открывающей скобки/кавычки ("(хлороза) листьев..." —
     * первый символ '(', но первая буква "х" строчная и это всё
     * ещё продолжение предложения, а не новый самостоятельный
     * заголовок).
     */
    val nextLineFirstLetter = nextNonBlankLine?.firstOrNull { it.isLetter() }
    if (nextLineFirstLetter != null && nextLineFirstLetter.isLowerCase()) {
        return false
    }

    return true
}

/*
 * Заголовок записи в этом документе встречается в тексте в ДВУХ
 * взаимоисключающих позициях (обе — известный побочный эффект
 * PDFTextStripper, см. комментарий у looksLikeStrayTitleLine):
 *
 * - "в хвосте своей же записи" (частый случай): Возбудитель ->
 *   ... -> Меры борьбы -> заголовок записи. Заголовок нужно
 *   применить ЗАДНИМ ЧИСЛОМ ко всему, что только что накопили.
 * - "перед разделами записи" (редкий случай — например, когда
 *   на одной странице заканчивается одна запись и сразу же
 *   начинается другая): заголовок предшествует своим разделам,
 *   и относится к тому, что будет накапливаться ПОСЛЕ него, а не
 *   к обрывку предыдущей записи, который уже накоплен до него.
 *
 * Локально (по одной этой строке) эти два случая неразличимы —
 * в обоих строка структурно выглядит одинаково. Различить их можно
 * только посмотрев, с чего НАЧАЛАСЬ уже накопленная последовательность
 * разделов (currentZone): если она начинается с раздела, которым
 * ЭМПИРИЧЕСКИ (см. discoverFirstSection) начинается любая запись
 * в этом документе — это самостоятельная, цельная запись, и найденная
 * строка — её собственный "уехавший" заголовок (первый случай).
 * Если зона начинается с чего-то другого (например, зона — это всего
 * один "хвостовой" раздел без начала) — это обрывок уже идущей записи,
 * и он остаётся под уже известным title, а найденная строка относится
 * к тому, что пойдёт ПОСЛЕ неё (второй случай).
 */
private fun splitIntoSectionBlocks(
    pages: List<Pair<Int, String>>,
    firstSectionMarker: String?
): List<SectionBlock> {

    val blocks = mutableListOf<SectionBlock>()

    var currentSection: String? = null
    var currentTitle: String? = null

    // Состояние текущей "зоны" — блоков со времени последнего
    // определённого title, ещё не привязанных окончательно.
    var zoneStartIndex = 0
    var zoneFirstSection: String? = null
    var seenSectionsInZone = mutableSetOf<String>()

    var buffer = StringBuilder()
    var bufferHasContent = false
    var previousLineEndedSentence = false

    fun flush(page: Int) {

        val text = buffer.toString().trim()

        if (text.isNotBlank()) {
            blocks += SectionBlock(
                page = page,
                section = currentSection,
                text = text,
                title = currentTitle
            )
        }

        buffer = StringBuilder()
        bufferHasContent = false
    }

    fun startNewZone(newTitle: String?) {
        currentTitle = newTitle
        zoneStartIndex = blocks.size
        zoneFirstSection = null
        seenSectionsInZone = mutableSetOf()
    }

    fun resolveZoneOnStrayTitle(newTitleLine: String) {
        if (zoneFirstSection != null && zoneFirstSection == firstSectionMarker) {
            for (i in zoneStartIndex until blocks.size) {
                blocks[i] = blocks[i].copy(title = newTitleLine)
            }
        }
        startNewZone(newTitleLine)
    }

    for ((page, pageText) in pages) {

        val lines = pageText.split('\n')

        for (index in lines.indices) {

            val rawLine = lines[index]
            val trimmedLine = rawLine.trim()
            val heading = detectSectionHeading(rawLine)

            if (heading != null) {
                flush(page)
                /*
                 * Раздел встречается в записи ровно один раз. Его
                 * повтор внутри одной зоны невозможен для настоящей
                 * записи — значит формально началась новая, даже
                 * если её собственный "уехавший" заголовок не пойман.
                 * Здесь у нас нет строки-кандидата на имя новой
                 * записи, поэтому title честно уходит в null, а не
                 * остаётся неверным, но правдоподобным.
                 */
                if (heading in seenSectionsInZone) {
                    startNewZone(null)
                }
                currentSection = heading
                if (zoneFirstSection == null) {
                    zoneFirstSection = heading
                }
                seenSectionsInZone += heading
                previousLineEndedSentence = false
                buffer.append(rawLine).append('\n')
                continue
            }

            if (
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
                flush(page)
                resolveZoneOnStrayTitle(trimmedLine)
                currentSection = null
            }

            buffer.append(rawLine).append('\n')

            if (trimmedLine.isNotEmpty()) {
                bufferHasContent = true
                previousLineEndedSentence = trimmedLine.last() in sentenceTerminators
            }
        }

        /*
         * chunk не должен растягиваться на две разные страницы —
         * поэтому буфер закрывается на границе страницы всегда,
         * но title-зона (currentTitle/zoneStartIndex/...) продолжается
         * и на следующей странице: запись может начинаться на одной
         * странице и продолжаться на следующей.
         */
        flush(page)
    }

    return blocks
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
