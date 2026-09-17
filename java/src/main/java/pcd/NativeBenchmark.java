package pcd;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import pcd.nativeengine.CompiledSchema;
import pcd.nativeengine.LlamaRuntime;
import pcd.nativeengine.NativeParallelEngine;

/**
 * Parallel constrained decoding natively in Java (libllama via FFM) — the counterpart of
 * {@code python -m core.benchmark}'s "Parallel Constrained" rows.
 *
 * <pre>
 *   java -jar target/pcd-benchmark.jar [-v] [--runs N] [--pad] [--single-pass] [preset.json ...]
 * </pre>
 *
 * Environment: {@code PCD_GGUF} (model path), {@code PCD_LLAMA_LIB_DIR} (libllama location).
 */
public final class NativeBenchmark {

    private static final String DEFAULT_GGUF = "models/qwen2.5-1.5b-instruct-q8_0.gguf";

    public static void main(String[] args) throws Exception {
        boolean verbose = false;
        boolean pad = false;
        boolean singlePass = false;
        int runs = 5;
        List<Path> presets = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-v", "--verbose" -> verbose = true;
                case "--pad" -> pad = true;
                case "--single-pass" -> singlePass = true;
                case "--runs" -> runs = Integer.parseInt(args[++i]);
                default -> presets.add(Path.of(args[i]));
            }
        }
        if (presets.isEmpty()) {
            presets = defaultPresets();
        }
        Path gguf = resolveModel();

        System.out.println("=".repeat(70));
        System.out.println("Parallel Constrained Decoding — native Java (libllama via FFM)");
        System.out.println("model: " + gguf + "  (median of " + runs + " runs after warm-up)");
        System.out.println("=".repeat(70));

        long t0 = System.nanoTime();
        try (LlamaRuntime rt = new LlamaRuntime(LlamaRuntime.Options.defaults(gguf))) {
            System.out.printf(Locale.ROOT, "Engine loaded in %.2fs.%n%n", (System.nanoTime() - t0) / 1e9);

            for (Path path : presets) {
                Preset preset = Preset.load(path);
                CompiledSchema schema = new CompiledSchema(rt, preset);
                NativeParallelEngine engine = new NativeParallelEngine(rt, schema, pad, singlePass);

                engine.run(preset, false); // warm-up: Metal shader compilation for these shapes
                NativeParallelEngine.Result cold = median(engine, preset, runs, false);
                NativeParallelEngine.Result warm = median(engine, preset, runs, true);

                System.out.printf("--> %s (%d fields, %d prompt tokens)%n",
                        preset.title(), preset.schema().size(), cold.promptTokens());
                print("cold (full prefill)      ", cold);
                print("warm (schema KV cached)  ", warm);
                if (!cold.multiLevelFields().isEmpty()) {
                    System.out.println("    >> multi-level fields: " + cold.multiLevelFields());
                }
                if (verbose) {
                    for (NativeParallelEngine.FieldResult f : cold.fields()) {
                        System.out.printf(Locale.ROOT, "    %-36s %-24s %.3f  (levels=%d)%n",
                                f.name(), f.value(), f.prob(), f.levels());
                    }
                }
                System.out.println("-".repeat(70));
            }
        }
    }

    private static NativeParallelEngine.Result median(NativeParallelEngine engine, Preset preset, int runs, boolean warm) {
        List<NativeParallelEngine.Result> all = new ArrayList<>();
        for (int i = 0; i < runs; i++) {
            all.add(engine.run(preset, warm));
        }
        all.sort((a, b) -> Double.compare(a.totalMs(), b.totalMs()));
        return all.get(all.size() / 2);
    }

    private static void print(String label, NativeParallelEngine.Result r) {
        System.out.printf(Locale.ROOT,
                "    %s: %7.1f ms | tokenize %4.1f | prefill %6.1f | broadcast %4.1f | suffix %5.1f | tree %5.1f (%d levels)%n",
                label, r.totalMs(), r.tokenizeMs(), r.prefillMs(), r.broadcastMs(), r.suffixMs(), r.treeMs(), r.treeLevels());
    }

    static Path resolveModel() {
        String env = System.getenv("PCD_GGUF");
        if (env != null) {
            return Path.of(env);
        }
        for (Path p : List.of(Path.of(DEFAULT_GGUF), Path.of("..", DEFAULT_GGUF))) {
            if (Files.exists(p)) {
                return p;
            }
        }
        throw new IllegalStateException("GGUF not found; set PCD_GGUF or place it at " + DEFAULT_GGUF);
    }

    private static List<Path> defaultPresets() throws Exception {
        Path dir = Files.isDirectory(Path.of("presets")) ? Path.of("presets") : Path.of("..", "presets");
        List<Path> out = new ArrayList<>();
        for (String id : List.of("fintech_fraud", "code_security", "support_triage", "high_cardinality_255")) {
            out.add(dir.resolve(id + ".json"));
        }
        return out;
    }
}
