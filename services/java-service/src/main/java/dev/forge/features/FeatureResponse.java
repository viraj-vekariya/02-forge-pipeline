package dev.forge.features;

import java.util.List;

/**
 * A validated, normalised vector plus the quality signals computed on the way.
 *
 * @param pixels        normalised values, ready for the model
 * @param stats         summary statistics of the input
 * @param warnings      non-fatal observations, e.g. an all-constant image
 * @param processMicros time spent, so the caller can attribute latency correctly
 */
public record FeatureResponse(
        List<Double> pixels,
        Stats stats,
        List<String> warnings,
        long processMicros) {

    public record Stats(double min, double max, double mean, double stdDev,
                        long zeroCount, double nonZeroFraction) {
    }
}
