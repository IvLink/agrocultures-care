package minirag.extensions

fun String.cleanJsonResponse(): String =
    trim()
        .removePrefix("```json")
        .removePrefix("```")
        .removeSuffix("```")
        .trim()
