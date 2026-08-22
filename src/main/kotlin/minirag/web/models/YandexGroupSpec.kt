package minirag.web.models

import kotlinx.serialization.Serializable

@Serializable
internal data class YandexGroupSpec(
    val groupsOnPage: String
)