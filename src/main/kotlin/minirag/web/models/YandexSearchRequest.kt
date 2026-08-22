package minirag.web.models

import kotlinx.serialization.Serializable

@Serializable
internal data class YandexSearchRequest(
    val query: YandexSearchQuery,
    val groupSpec: YandexGroupSpec,
    val region: String? = null,
    /*
     * Поле называется l10n в API (REST: CamelCase, поэтому l10n,
     * а не l10N/localization). По документации применяется только
     * к responseFormat=FORMAT_XML ("HTML: no" в таблице параметров),
     * поэтому при FORMAT_HTML (наш дефолт) остаётся не заданным.
     */
    val l10n: String? = null,
    val folderId: String,
    val responseFormat: String = "FORMAT_HTML"
)