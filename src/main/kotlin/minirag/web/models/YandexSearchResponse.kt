package minirag.web.models

import kotlinx.serialization.Serializable

@Serializable
internal data class YandexSearchResponse(
    val rawData: String? = null
)