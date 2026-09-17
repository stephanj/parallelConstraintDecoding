package pcd;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * One scenario: a context text plus the schema to extract from it, stored as presets/*.json.
 * A preset may also carry a bank of alternative inputs ({@code samples}) so the demo can pick a
 * different one each run; a sample can record the values a human assigned ({@code expected}).
 */
public record Preset(String id, String title, String description, String context, Map<String, FieldDef> schema,
        List<Sample> samples) {

    public record FieldDef(String name, String type, String description, List<String> choices) {
        public boolean isBoolean() {
            return "boolean".equals(type);
        }
    }

    public record Sample(String label, String context, Map<String, String> expected) {}

    public Preset(String id, String title, String description, String context, Map<String, FieldDef> schema) {
        this(id, title, description, context, schema, List.of());
    }

    public Preset withSamples(List<Sample> samples) {
        return new Preset(id, title, description, context, schema, List.copyOf(samples));
    }

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Pattern ID = Pattern.compile("[a-z0-9_]{1,64}");
    private static final Pattern FIELD_NAME = Pattern.compile("[A-Za-z_][A-Za-z0-9_]{0,63}");

    public static Preset load(Path path) throws IOException {
        return fromJson(MAPPER.readTree(Files.readString(path)));
    }

    /** Parses and validates the on-disk / API JSON shape; throws IllegalArgumentException on bad input. */
    public static Preset fromJson(JsonNode root) {
        String id = root.path("id").asText("");
        if (!ID.matcher(id).matches()) {
            throw new IllegalArgumentException("id must match [a-z0-9_]{1,64}, got '" + id + "'");
        }
        String context = root.path("context").asText("");
        if (context.isBlank()) {
            throw new IllegalArgumentException("context must not be empty");
        }
        JsonNode schemaNode = root.get("schema");
        if (schemaNode == null || !schemaNode.isObject() || schemaNode.isEmpty()) {
            throw new IllegalArgumentException("schema must be a non-empty object");
        }
        Map<String, FieldDef> schema = new LinkedHashMap<>();
        schemaNode.fields().forEachRemaining(e -> {
            String name = e.getKey();
            if (!FIELD_NAME.matcher(name).matches()) {
                throw new IllegalArgumentException("invalid field name '" + name + "'");
            }
            JsonNode spec = e.getValue();
            String type = spec.path("type").asText("enum");
            List<String> choices = new ArrayList<>();
            if ("boolean".equals(type)) {
                choices.addAll(List.of("true", "false"));
            } else if ("enum".equals(type)) {
                spec.path("choices").forEach(c -> {
                    String choice = c.asText().strip();
                    if (!choice.isEmpty() && !choices.contains(choice)) {
                        choices.add(choice);
                    }
                });
                if (choices.isEmpty()) {
                    throw new IllegalArgumentException("field '" + name + "' of type enum must have choices");
                }
                if (choices.size() > 255) {
                    throw new IllegalArgumentException("field '" + name + "' exceeds 255 choices");
                }
            } else {
                throw new IllegalArgumentException("field '" + name + "': unsupported type '" + type + "'");
            }
            schema.put(name, new FieldDef(name, type, spec.path("description").asText(""), List.copyOf(choices)));
        });
        List<Sample> samples = new ArrayList<>();
        for (JsonNode sn : root.path("samples")) {
            String sctx = sn.path("context").asText("");
            if (sctx.isBlank()) {
                continue;
            }
            Map<String, String> expected = new LinkedHashMap<>();
            sn.path("expected").fields().forEachRemaining(e -> expected.put(e.getKey(), e.getValue().asText()));
            samples.add(new Sample(sn.path("label").asText(""), sctx, Map.copyOf(expected)));
        }
        return new Preset(id, root.path("title").asText(id), root.path("description").asText(""), context, schema,
                List.copyOf(samples));
    }

    public ObjectNode toJson() {
        ObjectNode root = MAPPER.createObjectNode();
        root.put("id", id).put("title", title).put("description", description).put("context", context);
        ObjectNode schemaNode = root.putObject("schema");
        for (FieldDef f : schema.values()) {
            ObjectNode spec = schemaNode.putObject(f.name());
            spec.put("type", f.type()).put("description", f.description());
            if (!f.isBoolean()) {
                ArrayNode arr = spec.putArray("choices");
                f.choices().forEach(arr::add);
            }
        }
        if (!samples.isEmpty()) {
            ArrayNode arr = root.putArray("samples");
            for (Sample s : samples) {
                ObjectNode sn = arr.addObject();
                sn.put("label", s.label()).put("context", s.context());
                ObjectNode exp = sn.putObject("expected");
                s.expected().forEach(exp::put);
            }
        }
        return root;
    }

    public void save(Path path) throws IOException {
        Files.writeString(path, MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(toJson()) + "\n");
    }
}
