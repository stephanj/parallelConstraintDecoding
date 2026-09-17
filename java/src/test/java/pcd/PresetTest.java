package pcd;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.List;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;

class PresetTest {

    private static final ObjectMapper M = new ObjectMapper();

    private static Preset parse(String json) throws Exception {
        return Preset.fromJson(M.readTree(json));
    }

    @Test
    void roundTripKeepsFieldOrderChoicesAndBooleanShape() throws Exception {
        Preset p = parse("""
            {"id":"t1","title":"T","description":"d","context":"some text",
             "schema":{"zeta":{"type":"enum","description":"z","choices":["B","A"]},
                       "alpha":{"type":"boolean","description":"a"}}}""");

        assertEquals(List.of("zeta", "alpha"), List.copyOf(p.schema().keySet()));
        assertEquals(List.of("B", "A"), p.schema().get("zeta").choices());
        assertEquals(List.of("true", "false"), p.schema().get("alpha").choices());

        var json = p.toJson();
        assertFalse(json.get("schema").get("alpha").has("choices"), "booleans carry no choices on disk");
        assertEquals(p, Preset.fromJson(json));
    }

    @Test
    void dropsBlankAndDuplicateChoices() throws Exception {
        Preset p = parse("""
            {"id":"t","context":"x","schema":{"f":{"type":"enum","choices":[" A ","","A","B"]}}}""");
        assertEquals(List.of("A", "B"), p.schema().get("f").choices());
        assertEquals("t", p.title(), "title defaults to the id");
    }

    @Test
    void rejectsInvalidInput() {
        assertThrows(IllegalArgumentException.class, () -> parse("""
            {"id":"Bad-Id","context":"x","schema":{"f":{"type":"boolean"}}}"""));
        assertThrows(IllegalArgumentException.class, () -> parse("""
            {"id":"ok","context":"  ","schema":{"f":{"type":"boolean"}}}"""));
        assertThrows(IllegalArgumentException.class, () -> parse("""
            {"id":"ok","context":"x","schema":{}}"""));
        assertThrows(IllegalArgumentException.class, () -> parse("""
            {"id":"ok","context":"x","schema":{"f":{"type":"enum"}}}"""));
        assertThrows(IllegalArgumentException.class, () -> parse("""
            {"id":"ok","context":"x","schema":{"f":{"type":"string"}}}"""));
        assertThrows(IllegalArgumentException.class, () -> parse("""
            {"id":"ok","context":"x","schema":{"bad name":{"type":"boolean"}}}"""));
    }

    @Test
    void rejectsMoreThan255Choices() {
        String choices = IntStream.range(0, 256).mapToObj(i -> "\"C" + i + "\"").reduce((a, b) -> a + "," + b).get();
        assertThrows(IllegalArgumentException.class, () -> parse(
                "{\"id\":\"ok\",\"context\":\"x\",\"schema\":{\"f\":{\"type\":\"enum\",\"choices\":[" + choices + "]}}}"));
    }

    @Test
    void isBooleanReflectsType() throws Exception {
        Preset p = parse("""
            {"id":"t","context":"x","schema":{"b":{"type":"boolean"},"e":{"type":"enum","choices":["X"]}}}""");
        assertTrue(p.schema().get("b").isBoolean());
        assertFalse(p.schema().get("e").isBoolean());
    }
}
