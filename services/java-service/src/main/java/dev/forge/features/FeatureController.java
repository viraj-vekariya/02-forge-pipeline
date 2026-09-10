package dev.forge.features;

import jakarta.validation.Valid;
import java.util.LinkedHashMap;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/** HTTP surface for the feature service. */
@RestController
@RequestMapping("/api/features")
public class FeatureController {

    private final FeatureService service;

    public FeatureController(FeatureService service) {
        this.service = service;
    }

    @PostMapping("/normalise")
    public ResponseEntity<FeatureResponse> normalise(@Valid @RequestBody FeatureRequest request) {
        return ResponseEntity.ok(service.process(request.pixels()));
    }

    @GetMapping("/health")
    public Map<String, Object> health() {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("status", "ok");
        body.put("service", "feature-service");
        body.put("javaVersion", System.getProperty("java.version"));
        body.put("normalisationMean", service.normMean());
        body.put("normalisationStdDev", service.normStdDev());
        body.put("processed", service.processedCount());
        body.put("rejected", service.rejectedCount());
        body.put("warned", service.warnedCount());
        return body;
    }

    /**
     * Bad input is the CALLER's fault and must be a 4xx.
     *
     * <p>If it leaked out as a 500 it would count against the service's own error
     * budget and, in a system with circuit breakers, malformed client requests could
     * take a perfectly healthy service out of rotation.
     */
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Map<String, Object>> badInput(IllegalArgumentException ex) {
        return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                .body(Map.of("error", "invalid_input", "detail", ex.getMessage()));
    }
}
