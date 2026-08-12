package minirag.pdf

import org.apache.pdfbox.Loader
import org.apache.pdfbox.text.PDFTextStripper
import java.io.File

// ── ШАГ 1: PDF -> текст (готово, аналог pypdf.PdfReader) ─────────────────
fun readPdf(path: String): String {
    val document = Loader.loadPDF(File(path))
    return document.use { PDFTextStripper().getText(it) }
}
