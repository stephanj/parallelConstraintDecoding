package pcd.nativeengine;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import pcd.Preset;

@EnabledIfEnvironmentVariable(named = "PCD_TEST_GGUF", matches = ".+")
class GrammarNativeTest {
    @Test void constrainedGenerationMatchesPresetsAndHandlesEscapingAndTruncation() throws Exception {
        try (var rt = new LlamaRuntime(LlamaRuntime.Options.defaults(Path.of(System.getenv("PCD_TEST_GGUF"))))) {
            var engine = new GrammarJsonEngine(rt, 700);
            try (var files = Files.list(Path.of("../presets"))) {
                for (Path file : files.filter(p -> p.toString().endsWith(".json")).sorted().toList()) {
                    assertComplete(engine, Preset.load(file));
                }
            }
            // Singleton enums force exact escaping and UTF-8 across tokenizer byte boundaries.
            for (String value : List.of("a\"b\\c\n\t{}", "café 中文 😀")) {
                var fields = new LinkedHashMap<String, Preset.FieldDef>();
                fields.put("value", new Preset.FieldDef("value", "enum", "Exact value", List.of(value)));
                Preset preset = new Preset("escapes", "Escapes", "", "Return the required value", fields);
                assertComplete(engine, preset);
            }
            var shortResult = new GrammarJsonEngine(rt, 1).run(GrammarJsonEngineTest.preset(), s -> {});
            assertFalse(shortResult.completed());
            assertFalse(shortResult.output().schemaMatch());
            // A failed/truncated request must not leave sampler state in the next request.
            assertComplete(engine, GrammarJsonEngineTest.preset());
        }
    }

    private static void assertComplete(GrammarJsonEngine engine, Preset preset) {
        StringBuilder streamed = new StringBuilder();
        var result = engine.run(preset, streamed::append);
        assertTrue(result.completed(), preset.id());
        assertTrue(result.output().schemaMatch(), result.output().text());
        assertEquals(result.output().text(), streamed.toString());
        assertTrue(result.forwardPasses() > result.output().tokens());
        System.out.printf("grammar test: %s, %d tokens, %.1f ms%n", preset.id(),
                result.output().tokens(), result.output().elapsedMs());
    }
}
