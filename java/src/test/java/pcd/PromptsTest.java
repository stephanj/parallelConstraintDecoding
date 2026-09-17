package pcd;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;
import pcd.nativeengine.CompiledSchema;

class PromptsTest {

    private static Preset preset(Map<String, Preset.FieldDef> schema) {
        return new Preset("t", "t", "", "ctx", schema);
    }

    @Test
    void catalogListsDescriptionAndChoicesLikeThePythonEngine() {
        Map<String, Preset.FieldDef> schema = new LinkedHashMap<>();
        schema.put("risk_tier", new Preset.FieldDef("risk_tier", "enum", "Calculated risk tier\nmore", List.of("LOW", "HIGH")));
        schema.put("is_fraud", new Preset.FieldDef("is_fraud", "boolean", "Whether fraudulent", List.of("true", "false")));

        assertEquals(
                "  \"risk_tier\": Calculated risk tier [LOW | HIGH]\n  \"is_fraud\": Whether fraudulent [true | false]",
                Prompts.parallelCatalog(preset(schema)));
    }

    @Test
    void largeChoiceListsAreTruncatedToTwentyWithACount() {
        List<String> codes = IntStream.range(0, 255).mapToObj(i -> "C" + i).toList();
        Map<String, Preset.FieldDef> schema = new LinkedHashMap<>();
        schema.put("code", new Preset.FieldDef("code", "enum", "Code", codes));

        String line = Prompts.parallelCatalog(preset(schema));
        assertEquals(20, line.split(" \\| ").length - 1, "20 choices plus the overflow note");
        assertEquals(true, line.endsWith("| ... (255 total options)]"));
    }

    @Test
    void commonPrefixOfChoices() {
        assertEquals("CWE_", CompiledSchema.commonPrefix(List.of("CWE_79", "CWE_89", "CWE_502")));
        assertEquals("", CompiledSchema.commonPrefix(List.of("TIER_1_LOW", "SANCTIONED")));
        assertEquals("LOW", CompiledSchema.commonPrefix(List.of("LOW", "LOW_PRIORITY")));
    }
}
