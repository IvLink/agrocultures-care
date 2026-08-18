package minirag.pdf

import minirag.models.DocumentPage
import org.apache.pdfbox.Loader
import org.apache.pdfbox.text.PDFTextStripper
import java.io.File

/**
 * Извлекает текст постранично.
 *
 * getText(document) целиком не гарантирует form-feed ('')
 * между страницами у всех PDF, поэтому границы страниц раньше
 * терялись. Здесь страницы читаются по одной через
 * startPage/endPage, так что page у DocumentChunk соответствует
 * реальному номеру страницы в PDF.
 */
fun readPdfPages(path: String): List<DocumentPage> {
    val document = Loader.loadPDF(File(path))
    return document.use { doc ->
        val stripper = PDFTextStripper()
        (1..doc.numberOfPages).map { pageNumber ->
            stripper.startPage = pageNumber
            stripper.endPage = pageNumber
            DocumentPage(
                number = pageNumber,
                text = stripper.getText(doc)
            )
        }
    }
}
