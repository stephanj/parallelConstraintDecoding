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

    private LlamaRuntime rt;
    private GrammarJsonEngine naive;
    private final ReentrantLock lock = new ReentrantLock(true);
    /** Compiled schemas by their JSON shape; bounded LRU because ad-hoc callers (the Tetris view) send a new schema per turn. */
    private final Map<String, NativeParallelEngine> engines = new LinkedHashMap<>(64, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<String, NativeParallelEngine> eldest) {
            return size() > 64;
        }
    };
    private volatile Path modelPath;
    private volatile String modelId;

    EngineService(Path modelPath, String modelId) {
        load(modelPath, modelId);
    }

    Path modelPath() {
        return modelPath;
    }

    /** The catalog id of the loaded model (a file name, or {@code ollama:name:tag}). */
    String modelName() {
        return modelId;
    }

    /**
     * Replaces the loaded model. Waits for any run in progress; runs queued behind it see the new
     * model. If the new file fails to load (unsupported architecture, corrupt file), the previous
     * model is reloaded and the error propagates.
     */
    void switchModel(Path newModel, String newId) {
        lock.lock();
        try {
            if (newModel.equals(modelPath)) {
                return;
            }
            Path previous = modelPath;
            String previousId = modelId;
            rt.close();
            rt = null;
            engines.clear();
            try {
                load(newModel, newId);
            } catch (RuntimeException e) {
                load(previous, previousId);
                throw new IllegalArgumentException("cannot load " + newId + ": " + e.getMessage(), e);
            }
        } finally {
            lock.unlock();
        }
    }

    private void load(Path path, String id) {
        long t0 = System.nanoTime();
        rt = new LlamaRuntime(LlamaRuntime.Options.defaults(path));
        naive = new GrammarJsonEngine(rt, 700);
        modelPath = path;
        modelId = id;
        System.out.printf("Loaded %s (%s, %s) in %.1fs%n", id, rt.metaValue("general.architecture"),
                rt.hasChatTemplate() ? "chat template from file" : "no template, ChatML fallback", (System.nanoTime() - t0) / 1e9);
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
        return engines.computeIfAbsent(key, k -> {
            NativeParallelEngine engine = new NativeParallelEngine(rt,
                    new CompiledSchema(rt, preset, CompiledSchema.PromptStyle.SHARED), false, false);
            if (preset.schema().size() > 1) {
                engine.run(preset, false); // warm-up: Metal compiles kernels for this schema's batch shapes
            }
            return engine;
        });
    }

    private static double round(double v) {
        return Math.round(v * 1000.0) / 1000.0;
    }

    @Override
    public void close() {
        rt.close();
    }
}
