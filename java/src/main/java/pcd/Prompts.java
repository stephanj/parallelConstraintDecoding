package pcd;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Prompt text mirroring StructuredSchema.to_parallel_schema_str() in python/core/schema.py, so the
 * Java and Python engines give the model the same information.
 */
public final class Prompts {

    private Prompts() {}

    /** System prompt of the autoregressive baseline (python/core/prompt_builder.py::build_naive_json_prompt). */
    public static String naiveSystem(Preset preset) {
        return "You are a precise data extraction system. You must output ONLY a valid, beautifully formatted, "
                + "indented JSON object with newlines and 2-space indentation matching the schema below. "
                + "Do not output a single-line string. Do not include markdown tags.\n\n"
                + "JSON Schema:\n" + jsonSchemaCatalog(preset);
    }

    public static String naiveUser(Preset preset) {
        return "Analyze the following context and generate the required formatted JSON object:\n\n" + preset.context();
    }

    public static String grammarSystem(Preset preset) {
        return "You are a precise data extraction system. Output ONLY a compact JSON object "
                + "matching the schema below. Include every field exactly once in the listed order. "
                + "Do not include markdown or whitespace outside strings.\n\nJSON Schema:\n"
                + jsonSchemaCatalog(preset);
    }

    /** Mirrors StructuredSchema.to_json_schema_prompt_str(): a TypeScript-like schema listing. */
    public static String jsonSchemaCatalog(Preset preset) {
        StringBuilder sb = new StringBuilder("{\n");
        for (Preset.FieldDef f : preset.schema().values()) {
            if (f.isBoolean()) {
                sb.append("  \"").append(f.name()).append("\": boolean, // ").append(f.description()).append('\n');
            } else {
                String choices = truncatedChoices(f.choices()).stream()
                        .map(c -> "\"" + c + "\"")
                        .collect(Collectors.joining(" | "));
                sb.append("  \"").append(f.name()).append("\": ").append(choices)
                        .append(overflowNote(f.choices()))
                        .append(", // ").append(f.description()).append('\n');
            }
        }
        return sb.append('}').toString();
    }

    /** The schema catalog that is prefilled once (system prompt of the parallel engine). */
    public static String parallelSystem(Preset preset) {
        return "Classify JSON attributes:\n" + parallelCatalog(preset);
    }

    static String parallelCatalog(Preset preset) {
        return preset.schema().values().stream()
                .map(f -> {
                    String desc = f.description().split("\n")[0].strip();
                    String choices = f.isBoolean()
                            ? "true | false"
                            : String.join(" | ", truncatedChoices(f.choices())) + overflowNote(f.choices());
                    return "  \"" + f.name() + "\": " + desc + " [" + choices + "]";
                })
                .collect(Collectors.joining("\n"));
    }

    /** Same truncation rule as Python: all choices if <= 50, otherwise the first 20. */
    private static List<String> truncatedChoices(List<String> choices) {
        int limit = choices.size() > 50 ? 20 : choices.size();
        return choices.subList(0, limit);
    }

    private static String overflowNote(List<String> choices) {
        return choices.size() > 50 ? " | ... (" + choices.size() + " total options)" : "";
    }
}
