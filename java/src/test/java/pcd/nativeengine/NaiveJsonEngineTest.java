package pcd.nativeengine;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import pcd.Preset;

class NaiveJsonEngineTest {

    private static Preset preset() {
        Map<String, Preset.FieldDef> schema = new LinkedHashMap<>();
        schema.put("risk", new Preset.FieldDef("risk", "enum", "", List.of("LOW", "HIGH")));
        schema.put("block", new Preset.FieldDef("block", "boolean", "", List.of("true", "false")));
        return new Preset("t", "t", "", "ctx", schema);
    }

    @Test
    void validOutputMatchesSchema() {
        var r = NaiveJsonEngine.validate(preset(), "{\n  \"risk\": \"HIGH\",\n  \"block\": true\n}", 12, 100.0);
        assertTrue(r.validJson());
        assertTrue(r.schemaMatch());
        assertEquals(120.0, r.tokensPerSecond(), 1e-9);
    }

    @Test
    void reportsMissingKeysAndValuesOutsideTheEnum() {
        var r = NaiveJsonEngine.validate(preset(), "{\"risk\": \"MEDIUM\"}", 5, 50.0);
        assertTrue(r.validJson());
        assertFalse(r.schemaMatch());
        assertEquals(List.of("block"), r.missingKeys());
        assertEquals(List.of("risk=MEDIUM"), r.invalidEnums());
    }

    @Test
    void extraKeysDoNotBreakSchemaMatchOnTheirOwn() {
        var r = NaiveJsonEngine.validate(preset(), "{\"risk\": \"LOW\", \"block\": false, \"note\": \"x\"}", 5, 50.0);
        assertTrue(r.schemaMatch());
        assertTrue(r.parsed().has("note"));
    }

    @Test
    void brokenJsonIsRecordedNotThrown() {
        var r = NaiveJsonEngine.validate(preset(), "{\n  \"risk\": \"HIGH\",\n  \"block\": tr", 7, 50.0);
        assertFalse(r.validJson());
        assertFalse(r.schemaMatch());
        assertNull(r.parsed());
        assertTrue(r.missingKeys().isEmpty(), "no field checks on unparseable output");
    }

    @Test
    void objectIsExtractedFromSurroundingProseOrFences() {
        var r = NaiveJsonEngine.validate(preset(), "Here you go:\n```json\n{\"risk\": \"LOW\", \"block\": false}\n```", 9, 50.0);
        assertTrue(r.validJson());
        assertTrue(r.schemaMatch());
    }
}
