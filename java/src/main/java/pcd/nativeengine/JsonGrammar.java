package pcd.nativeengine;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.stream.Collectors;
import pcd.Preset;

/** Exact compact JSON grammar for the project's flat boolean/enum schemas. */
public final class JsonGrammar {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private JsonGrammar() {}

    public static String compile(Preset preset) {
        StringBuilder root = new StringBuilder("root ::= \"{\"");
        StringBuilder rules = new StringBuilder();
        int i = 0;
        for (Preset.FieldDef field : preset.schema().values()) {
            if (i > 0) root.append(" \",\"");
            root.append(' ').append(literal(json(field.name()) + ":")).append(" value-").append(i);
            rules.append("value-").append(i++).append(" ::= ");
            rules.append(field.isBoolean() ? "\"true\" | \"false\""
                    : field.choices().stream().map(JsonGrammar::json).map(JsonGrammar::literal)
                            .collect(Collectors.joining(" | ")));
            rules.append('\n');
        }
        return root.append(" \"}\"\n").append(rules).toString();
    }

    private static String json(String value) {
        try {
            return MAPPER.writeValueAsString(value);
        } catch (JsonProcessingException e) {
            throw new IllegalArgumentException("Cannot encode JSON string", e);
        }
    }

    // JSON quoting also escapes the quotes/backslashes of each GBNF terminal.
    private static String literal(String value) {
        return json(value);
    }
}
