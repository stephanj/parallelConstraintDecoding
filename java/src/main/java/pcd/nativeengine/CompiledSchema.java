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

    /** Which prompt the prefix carries: the Python port's one-line catalog, or the same prompt as the baselines. */
    public enum PromptStyle { CATALOG, SHARED }

    private static final String USER_MARKER = "\u0001PCD-USER-CONTENT\u0001";

    public final PromptStyle promptStyle;
    public final String systemPrompt;
    /** What follows the user content in the template: end of user turn + assistant turn opener. */
    public final String userSuffix;
    public final int[] systemTokens;
    public final List<Field> fields;
    /** Tokens that can legally follow a completed value: {@code "} possibly merged with {@code ,} / newline. */
    public final int[] closingTokens;

    public CompiledSchema(LlamaRuntime rt, Preset preset) {
        this(rt, preset, PromptStyle.CATALOG);
    }

    public CompiledSchema(LlamaRuntime rt, Preset preset, PromptStyle promptStyle) {
        this.promptStyle = promptStyle;
        String system = promptStyle == PromptStyle.SHARED ? Prompts.grammarSystem(preset) : Prompts.parallelSystem(preset);
        // Render the model's own chat template once with a marker as the user content, then split:
        // everything before the marker is the cacheable prefix, everything after it (the end of the
        // user turn and the opening of the assistant turn) is appended to each context.
        String rendered = rt.chatPrompt(system, USER_MARKER);
        int at = rendered.indexOf(USER_MARKER);
        if (at < 0) {
            throw new IllegalStateException("chat template did not place the user content verbatim");
        }
        systemPrompt = rendered.substring(0, at);
        userSuffix = rendered.substring(at + USER_MARKER.length());
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

    /** The user turn for a given context, matching the baseline's wording when the prompt is shared. */
    public String userTurn(Preset preset) {
        return promptStyle == PromptStyle.SHARED ? Prompts.naiveUser(preset) : preset.context();
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
