package pcd.nativeengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import pcd.Preset;
import pcd.Prompts;

/**
 * The autoregressive baseline on libllama: ask the model to write the whole JSON object and decode
 * it greedily, one token per forward pass, streaming each piece to a callback. Mirrors
 * python/core/engine_mlx.py::run_naive_generation, including the {@code {\n  } assistant prefill.
 */
public final class NaiveJsonEngine {

    public record Result(
            double elapsedMs,
            int tokens,
            String text,
            JsonNode parsed,
            boolean validJson,
            List<String> missingKeys,
            List<String> invalidEnums) {

        public boolean schemaMatch() {
            return validJson && missingKeys.isEmpty() && invalidEnums.isEmpty();
        }

        public double tokensPerSecond() {
            return elapsedMs > 0 ? tokens / (elapsedMs / 1000.0) : 0;
        }
    }

    private static final Pattern OBJECT = Pattern.compile("(\\{.*\\})", Pattern.DOTALL);
    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final String PREFILL = "{\n  ";

    private final LlamaRuntime rt;
    private final int maxTokens;
    private final float[] scratch;

    public NaiveJsonEngine(LlamaRuntime rt, int maxTokens) {
        this.rt = rt;
        this.maxTokens = maxTokens;
        this.scratch = new float[rt.vocabSize()];
    }

    /** Generates the JSON for {@code preset}; {@code onPiece} receives each decoded piece as it is produced. */
    public Result run(Preset preset, Consumer<String> onPiece) {
        String prompt = "<|im_start|>system\n" + Prompts.naiveSystem(preset) + "<|im_end|>\n"
                + "<|im_start|>user\n" + Prompts.naiveUser(preset) + "<|im_end|>\n"
                + "<|im_start|>assistant\n" + PREFILL;
        int[] promptTokens = rt.tokenize(prompt, true);

        long t0 = System.nanoTime();
        rt.clearMemory();
        List<LlamaRuntime.Tok> batch = new ArrayList<>(promptTokens.length);
        for (int i = 0; i < promptTokens.length; i++) {
            batch.add(new LlamaRuntime.Tok(promptTokens[i], 0, i, i == promptTokens.length - 1));
        }
        int out = rt.decode(batch).get(0);
        int next = rt.argmax(out, scratch);

        StringBuilder text = new StringBuilder(PREFILL);
        onPiece.accept(PREFILL);
        int pos = promptTokens.length;
        int generated = 0;
        while (generated < maxTokens && !rt.isEndOfGeneration(next)) {
            String piece = rt.piece(next);
            text.append(piece);
            onPiece.accept(piece);
            generated++;
            if (balanced(text)) {
                break;
            }
            out = rt.decode(List.of(new LlamaRuntime.Tok(next, 0, pos++, true))).get(0);
            next = rt.argmax(out, scratch);
        }
        double elapsedMs = (System.nanoTime() - t0) / 1_000_000.0;
        return validate(preset, text.toString(), generated, elapsedMs);
    }

    /** True once the text ends with '}' and braces balance — the object is complete. */
    private static boolean balanced(CharSequence text) {
        int depth = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c == '{') {
                depth++;
            } else if (c == '}') {
                depth--;
            }
        }
        return depth == 0 && text.toString().stripTrailing().endsWith("}");
    }

    static Result validate(Preset preset, String text, int tokens, double elapsedMs) {
        Matcher m = OBJECT.matcher(text.strip());
        String candidate = m.find() ? m.group(1) : text.strip();
        JsonNode parsed = null;
        boolean valid = false;
        try {
            parsed = MAPPER.readTree(candidate);
            valid = parsed != null && parsed.isObject();
        } catch (Exception ignored) {
            // invalid JSON is a legitimate outcome of the baseline, recorded as validJson=false
        }
        List<String> missing = new ArrayList<>();
        List<String> invalid = new ArrayList<>();
        if (valid) {
            for (Preset.FieldDef f : preset.schema().values()) {
                JsonNode v = parsed.get(f.name());
                if (v == null) {
                    missing.add(f.name());
                } else if (!f.isBoolean() && !f.choices().contains(v.asText())) {
                    invalid.add(f.name() + "=" + v.asText());
                }
            }
        }
        return new Result(elapsedMs, tokens, text, parsed, valid, missing, invalid);
    }
}
