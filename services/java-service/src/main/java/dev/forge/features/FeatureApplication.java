package dev.forge.features;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Feature service - validation and normalisation ahead of the model.
 *
 * <p>Built, tested, containerised and deployed by the same pipeline as the Python
 * inference service. Nothing in the pipeline is aware that one is Python and the other
 * is Java, which is the property the project exists to demonstrate.
 */
@SpringBootApplication
public class FeatureApplication {
    public static void main(String[] args) {
        SpringApplication.run(FeatureApplication.class, args);
    }
}
