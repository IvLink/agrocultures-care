package minirag.extensions

import kotlin.math.sqrt

fun List<Double>.norm(): Double = sqrt(sumOf { it * it })

fun List<Double>.cosineSimilarity(other: List<Double>): Double =
    zip(other) { a, b -> a * b }.sum() / (norm() * other.norm() + 1e-9)
