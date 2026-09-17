package pcd.web;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import pcd.Preset;
import pcd.nativeengine.CompiledSchema;
import pcd.nativeengine.LlamaRuntime;
import pcd.nativeengine.NaiveJsonEngine;
import pcd.nativeengine.GrammarJsonEngine;
import pcd.nativeengine.NativeParallelEngine;

/**
 * Owns the single libllama runtime for the web server. libllama contexts are not thread-safe, so
 * every run goes through one lock; compiled schemas are cached by their JSON shape.
 */
final class EngineService implements AutoCloseable {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final LlamaRuntime rt;
    private final GrammarJsonEngine naive;
    private final ReentrantLock lock = new ReentrantLock(true);
    private final Map<String, NativeParallelEngine> engines = new LinkedHashMap<>();
    final Path modelPath;

    EngineService(Path modelPath) {
        this.modelPath = modelPath;
        this.rt = new LlamaRuntime(LlamaRuntime.Options.defaults(modelPath));
        this.naive = new GrammarJsonEngine(rt, 700);
    }

    /** Runs the parallel constrained engine; the result JSON is what the UI renders. */
    ObjectNode runParallel(Preset preset) {
        lock.lock();
        try {
            NativeParallelEngine engine = engineFor(preset);
            NativeParallelEngine.Result r = engine.run(preset, false);
            ObjectNode out = MAPPER.createObjectNode();
            out.put("elapsedMs", round(r.totalMs()));
            ObjectNode phases = out.putObject("phases");
            phases.put("tokenize", round(r.tokenizeMs()));
            phases.put("prefill", round(r.prefillMs()));
            phases.put("broadcast", round(r.broadcastMs()));
            phases.put("suffix", round(r.suffixMs()));
            phases.put("tree", round(r.treeMs()));
            out.put("promptTokens", r.promptTokens());
            out.put("treeLevels", r.treeLevels());
            out.put("forwardPasses", 2 + r.treeLevels());
            ObjectNode json = out.putObject("json");
            ArrayNode fields = out.putArray("fields");
            for (NativeParallelEngine.FieldResult f : r.fields()) {
                if (f.value() instanceof Boolean b) {
                    json.put(f.name(), b);
                } else {
                    json.put(f.name(), String.valueOf(f.value()));
                }
                ObjectNode fn = fields.addObject();
                fn.put("name", f.name());
                fn.put("value", String.valueOf(f.value()));
                fn.put("prob", round(f.prob()));
                fn.put("levels", f.levels());
                ObjectNode probs = fn.putObject("probs");
                f.probs().entrySet().stream()
                        .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                        .limit(6)
                        .forEach(e -> probs.put(e.getKey(), round(e.getValue())));
            }
            return out;
        } finally {
            lock.unlock();
        }
    }

    /** Runs the autoregressive baseline, streaming pieces to {@code onPiece}. */
    ObjectNode runNaive(Preset preset, Consumer<String> onPiece) {
        lock.lock();
        try {
            GrammarJsonEngine.Result generation = naive.run(preset, onPiece);
            NaiveJsonEngine.Result r = generation.output();
            ObjectNode out = MAPPER.createObjectNode();
            out.put("mode", "grammar_constrained_autoregressive");
            out.put("completed", generation.completed());
            out.put("elapsedMs", round(r.elapsedMs()));
            out.put("tokens", r.tokens());
            out.put("tokensPerSecond", round(r.tokensPerSecond()));
            out.put("forwardPasses", generation.forwardPasses());
            out.put("text", r.text());
            out.put("validJson", r.validJson());
            out.put("schemaMatch", r.schemaMatch());
            out.set("json", r.parsed() == null ? MAPPER.nullNode() : r.parsed());
            ArrayNode missing = out.putArray("missingKeys");
            r.missingKeys().forEach(missing::add);
            ArrayNode invalid = out.putArray("invalidEnums");
            r.invalidEnums().forEach(invalid::add);
            ArrayNode extra = out.putArray("extraKeys");
            if (r.parsed() != null && r.parsed().isObject()) {
                r.parsed().fieldNames().forEachRemaining(k -> {
                    if (!preset.schema().containsKey(k)) {
                        extra.add(k);
                    }
                });
            }
            return out;
        } finally {
            lock.unlock();
        }
    }

    private NativeParallelEngine engineFor(Preset preset) {
        String key = preset.toJson().get("schema").toString();
        return engines.computeIfAbsent(key, k -> new NativeParallelEngine(rt, new CompiledSchema(rt, preset), false, false));
    }

    private static double round(double v) {
        return Math.round(v * 1000.0) / 1000.0;
    }

    @Override
    public void close() {
        rt.close();
    }
}
