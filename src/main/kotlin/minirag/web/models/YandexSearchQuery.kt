package minirag.web.models

import kotlinx.serialization.Serializable

@Serializable
internal data class YandexSearchQuery(
    val searchType: String,
    val queryText: String,
    // Значения enum передаются полным именем ("FAMILY_MODE_STRICT"),
    // не коротким ("strict") — см. таблицу параметров в документации.
    val familyMode: String = "FAMILY_MODE_STRICT"
)