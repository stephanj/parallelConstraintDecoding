package pcd.nativeengine;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import pcd.Preset;
import pcd.Prompts;

/**
 * Per-schema token metadata, computed once and reused across calls (mirrors
 * StructuredSchema.compile_parallel_metadata in the Python engine).
 *
 * <p>For every field: the JSON suffix {@code   "name": "PREFIX} where PREFIX is the common prefix of
 * all choices, and for every choice the token sequence of its remainder after PREFIX. Suffix and
 * remainders are tokenized separately so BPE merges never cross the decision boundary.
 */
public final class CompiledSchema {

    public record Field(Preset.FieldDef def, int[] suffixTokens, String prefix, List<int[]> choiceTokens) {}

    public final String systemPrompt;
    public final int[] systemTokens;
    public final List<Field> fields;
    /** Tokens that can legally follow a completed value: {@code "} possibly merged with {@code ,} / newline. */
    public final int[] closingTokens;

    public CompiledSchema(LlamaRuntime rt, Preset preset) {
        systemPrompt = "<|im_start|>system\n" + Prompts.parallelSystem(preset) + "<|im_end|>\n<|im_start|>user\n";
        systemTokens = rt.tokenize(systemPrompt, true);

        Set<Integer> closing = new LinkedHashSet<>();
        for (String s : List.of("\"", "\",", "\"\n", "\",\n")) {
            int[] t = rt.tokenize(s, false);
            if (t.length > 0) {
                closing.add(t[0]);
            }
        }
        closingTokens = closing.stream().mapToInt(Integer::intValue).toArray();

        fields = new ArrayList<>();
        for (Preset.FieldDef def : preset.schema().values()) {
            String prefix = def.isBoolean() ? "" : commonPrefix(def.choices());
            String suffix = def.isBoolean() ? "  \"" + def.name() + "\": " : "  \"" + def.name() + "\": \"" + prefix;
            List<int[]> choiceTokens = new ArrayList<>();
            for (String choice : def.choices()) {
                choiceTokens.add(rt.tokenize(choice.substring(prefix.length()), false));
            }
            fields.add(new Field(def, rt.tokenize(suffix, false), prefix, choiceTokens));
        }
    }

    public static String commonPrefix(List<String> strings) {
        String prefix = strings.get(0);
        for (String s : strings) {
            int i = 0;
            while (i < prefix.length() && i < s.length() && prefix.charAt(i) == s.charAt(i)) {
                i++;
            }
            prefix = prefix.substring(0, i);
        }
        return prefix;
    }
}
