package pcd.nativeengine;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import org.junit.jupiter.api.Test;
import pcd.Preset;

class GrammarJsonEngineTest {
    static Preset preset() {
        var fields = new LinkedHashMap<String, Preset.FieldDef>();
        fields.put("flag", new Preset.FieldDef("flag", "boolean", "Whether true", List.of("true", "false")));
        fields.put("label", new Preset.FieldDef("label", "enum", "Label", List.of("YES", "NO")));
        return new Preset("test", "Test", "", "Choose YES and true", fields);
    }

    @Test void grammarRequiresTypedValuesAndAllKeys() {
        assertEquals("root ::= \"{\" \"\\\"flag\\\":\" value-0 \",\" \"\\\"label\\\":\" value-1 \"}\"\n"
                + "value-0 ::= \"true\" | \"false\"\n"
                + "value-1 ::= \"\\\"YES\\\"\" | \"\\\"NO\\\"\"\n", JsonGrammar.compile(preset()));
    }

    @Test void validatorRejectsWrongTypesAndExtraKeys() {
        for (String text : List.of("{\"flag\":\"true\",\"label\":\"YES\"}",
                "{\"flag\":null,\"label\":\"YES\"}", "{\"flag\":true,\"label\":12}",
                "{\"flag\":true,\"label\":\"YES\",\"extra\":0}")) {
            assertFalse(NaiveJsonEngine.validate(preset(), text, 0, 0).schemaMatch(), text);
        }
        assertTrue(NaiveJsonEngine.validate(preset(), "{\"flag\":true,\"label\":\"YES\"}", 0, 0).schemaMatch());
    }

    @Test void streamingWaitsForCompleteUnicodeCharacters() {
        byte[] bytes = "a😀".getBytes(StandardCharsets.UTF_8);
        for (int length = 2; length < bytes.length; length++) {
            assertEquals(1, GrammarJsonEngine.completeUtf8End(java.util.Arrays.copyOf(bytes, length)));
        }
        assertEquals(bytes.length, GrammarJsonEngine.completeUtf8End(bytes));
    }
}
