plugins {
    kotlin("jvm") version "2.4.10"
    kotlin("plugin.serialization") version "2.4.10"
    application
}

repositories {
    mavenCentral()
}

val ktorVersion = "3.5.2"

dependencies {
    implementation("io.ktor:ktor-client-core:$ktorVersion")
    implementation("io.ktor:ktor-client-cio:$ktorVersion")
    implementation("io.ktor:ktor-client-content-negotiation:$ktorVersion")
    implementation("io.ktor:ktor-serialization-kotlinx-json:$ktorVersion")
    implementation("org.apache.pdfbox:pdfbox:3.0.8") // JVM-аналог pypdf
    implementation("org.slf4j:slf4j-simple:2.0.18")
    implementation("com.microsoft.onnxruntime:onnxruntime:1.28.0")
    implementation("ai.djl.huggingface:tokenizers:0.36.0")
    implementation("ai.koog:koog-agents:1.0.0")
    implementation(kotlin("reflect"))
}

application {
    mainClass.set("minirag.MainKt")
    applicationDefaultJvmArgs = listOf("-Dstdout.encoding=UTF-8", "-Dstderr.encoding=UTF-8")
}

tasks.named<JavaExec>("run") {
    standardInput = System.`in`
}