package dev.forge.features;

import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import java.util.List;

/**
 * An incoming raw feature vector.
 *
 * <p>The size constraint is enforced here, at the edge, rather than inside the model
 * service. A malformed vector that reaches a traced TorchScript graph produces a shape
 * error deep in the stack, which surfaces as a 500 and reads as a model failure. Caught
 * here it is a 400 with a message naming the actual problem.
 */
public record FeatureRequest(
        @NotNull @Size(min = 784, max = 784, message = "expected exactly 784 pixels")
        List<Double> pixels) {
}
