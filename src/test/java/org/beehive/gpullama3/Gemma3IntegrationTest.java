package org.beehive.gpullama3;

import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.loader.ModelLoader;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Integration test for Gemma3 model loading (both normal and LiteRT GGUF variants).
 *
 * <p>The path to the Gemma3 GGUF model is read from {@code test.properties} on the classpath.
 * The test is skipped automatically when the configured file does not exist, so it remains
 * safe to run in environments where the model is not available (e.g. CI without model files).</p>
 *
 * <p>Supported model variants:
 * <ul>
 *   <li><b>Normal Gemma3</b> — standard GGUF produced by llama.cpp quantization; stores
 *       architecture metadata under the {@code gemma.*} key prefix.</li>
 *   <li><b>LiteRT Gemma3</b> — GGUF derived from Google's LiteRT-optimised checkpoints;
 *       stores architecture metadata under the {@code llama.*} key prefix, but the
 *       {@code general.architecture} field is {@code "gemma3"}.</li>
 * </ul>
 * </p>
 */
public class Gemma3IntegrationTest {

    private static final String PROPERTIES_FILE = "test.properties";
    private static final String MODEL_PATH_KEY = "gemma3.model.path";

    private static Path modelPath;

    @BeforeClass
    public static void loadProperties() throws IOException {
        Properties props = new Properties();
        try (InputStream in = Gemma3IntegrationTest.class.getClassLoader().getResourceAsStream(PROPERTIES_FILE)) {
            if (in != null) {
                props.load(in);
            }
        }

        String pathValue = props.getProperty(MODEL_PATH_KEY);
        if (pathValue != null && !pathValue.isBlank()) {
            modelPath = Path.of(pathValue);
        } else {
            System.out.println("[Gemma3IntegrationTest] " + MODEL_PATH_KEY + " not set in "
                    + PROPERTIES_FILE + " — test will be skipped.");
        }
    }

    /**
     * Verifies that the Gemma3 model file can be loaded and is detected as {@link ModelType#GEMMA_3}.
     * The test is skipped when the configured model file does not exist.
     */
    @Test
    public void testGemma3ModelLoads() throws IOException {
        Assume.assumeNotNull("gemma3.model.path not configured in test.properties", modelPath);
        Assume.assumeTrue("Gemma3 model file not found: " + modelPath, Files.exists(modelPath));

        Model model = ModelLoader.loadModel(modelPath, 512, true, false);

        assertNotNull("Loaded model must not be null", model);
        assertEquals("Model type must be GEMMA_3", ModelType.GEMMA_3, model.getModelType());
        assertNotNull("Model tokenizer must not be null", model.tokenizer());
        assertNotNull("Model configuration must not be null", model.configuration());
    }
}
