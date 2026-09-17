package pcd.nativeengine;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import pcd.Preset;
import pcd.Prompts;

/** Autoregressive baseline with an exact JSON grammar, using the same model/context. */
public final class GrammarJsonEngine {
    public record Result(NaiveJsonEngine.Result output, int forwardPasses, boolean completed) {}
    private final LlamaRuntime rt;
    private final int maxTokens;

    public GrammarJsonEngine(LlamaRuntime rt, int maxTokens) {
        if (maxTokens < 1) throw new IllegalArgumentException("maxTokens must be positive");
        this.rt = rt;
        this.maxTokens = maxTokens;
    }

    public Result run(Preset preset, Consumer<String> onPiece) {
        // Grammar construction is included in elapsed time; no hidden per-request setup.
        long start = System.nanoTime();
        String prompt = rt.chatPrompt(Prompts.grammarSystem(preset), Prompts.naiveUser(preset));
        int[] tokens = rt.tokenize(prompt, true);
        int passes = 0;
        int generated = 0;
        boolean completed = false;
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        int streamed = 0;
        try (var sampler = rt.grammarSampler(JsonGrammar.compile(preset))) {
            rt.clearMemory();
            int output = -1;
            for (int offset = 0; offset < tokens.length; offset += rt.batchCapacity()) {
                List<LlamaRuntime.Tok> batch = new ArrayList<>();
                int end = Math.min(tokens.length, offset + rt.batchCapacity());
                for (int i = offset; i < end; i++) {
                    batch.add(new LlamaRuntime.Tok(tokens[i], 0, i, i == tokens.length - 1));
                }
                List<Integer> outputs = rt.decode(batch);
                passes++;
                if (!outputs.isEmpty()) output = outputs.get(0);
            }
            int pos = tokens.length;
            while (true) {
                int next = sampler.sample(output);
                if (rt.isEndOfGeneration(next)) {
                    completed = true;
                    break;
                }
                if (generated == maxTokens) break;
                byte[] piece = rt.pieceBytes(next);
                bytes.writeBytes(piece);
                generated++;
                // Tokens may end partway through a UTF-8 character. Only stream complete bytes.
                byte[] all = bytes.toByteArray();
                int end = completeUtf8End(all);
                if (end > streamed) {
                    onPiece.accept(new String(all, streamed, end - streamed, StandardCharsets.UTF_8));
                    streamed = end;
                }
                output = rt.decode(List.of(new LlamaRuntime.Tok(next, 0, pos++, true))).get(0);
                passes++;
            }
        }
        var validated = NaiveJsonEngine.validate(preset, bytes.toString(StandardCharsets.UTF_8), generated,
                (System.nanoTime() - start) / 1_000_000.0);
        return new Result(validated, passes, completed);
    }

    static int completeUtf8End(byte[] bytes) {
        int start = bytes.length - 1;
        while (start >= 0 && (bytes[start] & 0xc0) == 0x80) start--;
        if (start < 0) return 0;
        int lead = bytes[start] & 0xff;
        int length = lead < 0x80 ? 1 : lead < 0xe0 ? 2 : lead < 0xf0 ? 3 : 4;
        return bytes.length - start < length ? start : bytes.length;
    }
}
