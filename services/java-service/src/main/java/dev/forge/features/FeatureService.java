package dev.forge.features;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import org.springframework.stereotype.Service;

/**
 * Normalisation and input quality checks.
 *
 * <p>The normalisation constants MUST match the ones the model was trained with. They
 * are read from configuration rather than hard-coded for exactly that reason: if
 * training changes them, this service is redeployed with the new values, and the
 * mismatch is a config diff rather than a silent accuracy collapse that looks like a
 * model regression.
 */
@Service
public class FeatureService {

    private final double mean;
    private final double stdDev;

    private final AtomicLong processed = new AtomicLong();
    private final AtomicLong rejected = new AtomicLong();
    private final AtomicLong warned = new AtomicLong();

    public FeatureService(
            @org.springframework.beans.factory.annotation.Value("${forge.norm.mean:0.2860}") double mean,
            @org.springframework.beans.factory.annotation.Value("${forge.norm.std:0.3530}") double stdDev) {
        if (stdDev <= 0) {
            throw new IllegalArgumentException("normalisation stdDev must be positive");
        }
        this.mean = mean;
        this.stdDev = stdDev;
    }

    public FeatureResponse process(List<Double> raw) {
        long started = System.nanoTime();

        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;
        double sum = 0;
        long zeros = 0;

        for (Double boxed : raw) {
            if (boxed == null || boxed.isNaN() || boxed.isInfinite()) {
                rejected.incrementAndGet();
                throw new IllegalArgumentException(
                        "pixel values must be finite numbers; found " + boxed);
            }
            double v = boxed;
            if (v < 0 || v > 255) {
                rejected.incrementAndGet();
                throw new IllegalArgumentException(
                        "pixel values must be in [0,255]; found " + v);
            }
            min = Math.min(min, v);
            max = Math.max(max, v);
            sum += v;
            if (v == 0) {
                zeros++;
            }
        }

        int n = raw.size();
        double avg = sum / n;

        // Two-pass variance rather than the sum-of-squares shortcut. The one-pass form
        // subtracts two large, nearly equal numbers and loses precision exactly when
        // the values are large and the variance is small - which is the case here.
        double sqDiff = 0;
        for (Double v : raw) {
            sqDiff += Math.pow(v - avg, 2);
        }
        double sd = Math.sqrt(sqDiff / n);

        List<String> warnings = new ArrayList<>();
        if (sd < 1e-6) {
            warnings.add("input is constant (stdDev " + sd + "); the model will "
                    + "produce a prediction but it carries no information");
        }
        if (zeros == n) {
            warnings.add("input is entirely zero");
        }
        if (max <= 1.0 && n > 0) {
            // A very common integration bug: the caller normalised to 0..1 already, so
            // normalising again shrinks everything toward zero and accuracy collapses
            // for reasons that look like a model problem.
            warnings.add("all values <= 1.0; input may already be scaled to 0..1 - "
                    + "double normalisation would silently degrade predictions");
        }
        if (!warnings.isEmpty()) {
            warned.incrementAndGet();
        }

        List<Double> normalised = new ArrayList<>(n);
        for (Double v : raw) {
            normalised.add(((v / 255.0) - mean) / stdDev);
        }

        processed.incrementAndGet();
        return new FeatureResponse(
                normalised,
                new FeatureResponse.Stats(min, max, round(avg), round(sd), zeros,
                        round(1.0 - (double) zeros / n)),
                warnings,
                (System.nanoTime() - started) / 1_000);
    }

    private static double round(double v) {
        return Math.round(v * 10_000.0) / 10_000.0;
    }

    public long processedCount() { return processed.get(); }
    public long rejectedCount()  { return rejected.get(); }
    public long warnedCount()    { return warned.get(); }
    public double normMean()     { return mean; }
    public double normStdDev()   { return stdDev; }
}
