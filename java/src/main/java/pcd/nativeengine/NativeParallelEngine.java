package pcd.nativeengine;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import pcd.Preset;

/**
 * Parallel constrained decoding on libllama — the same technique as core/engine_mlx.py, in Java:
 *
 * <ol>
 *   <li>prefill the ChatML prompt (schema catalog + context + {@code {\n}) once into sequence 0;
 *   <li>broadcast that KV prefix to one sequence per field with {@code llama_memory_seq_cp} (zero
 *       copy under kv_unified);
 *   <li>decode every field's suffix {@code   "name": "PREFIX} in a single batch and read the logits
 *       at each field's last suffix token;
 *   <li>score the allowed choices from those logits only (sub-vocabulary softmax);
 *   <li>when several choices share the winning first token, resolve them level by level: one
 *       batched decode per level for the still-ambiguous fields, so the confidence is the product of
 *       calibrated per-level probabilities — no fabricated fallback values;
 *   <li>assemble the JSON programmatically.
 * </ol>
 */
public final class NativeParallelEngine {

    public record FieldResult(String name, Object value, double prob, int levels, Map<String, Double> probs) {}

    public record Result(
            double totalMs,
            double tokenizeMs,
            double prefillMs,
            double broadcastMs,
            double suffixMs,
            double treeMs,
            int promptTokens,
            int treeLevels,
            List<FieldResult> fields) {

        public List<String> multiLevelFields() {
            return fields.stream().filter(f -> f.levels() > 1).map(FieldResult::name).toList();
        }
    }

    private static final int MAX_LEVELS = 24;

    private final LlamaRuntime rt;
    private final CompiledSchema schema;
    private final boolean padSuffixes;
    private final boolean singlePass;
    private int cachedSystemTokens = -1;

    /**
     * @param padSuffixes right-pad every suffix to the longest one (as the Python engine does)
     * @param singlePass tag the prefix tokens with every field's sequence and append the suffixes
     *     in the same llama_decode, so prefill + broadcast + suffix evaluation is one forward pass
     */
    public NativeParallelEngine(LlamaRuntime rt, CompiledSchema schema, boolean padSuffixes, boolean singlePass) {
        this.rt = rt;
        this.schema = schema;
        this.padSuffixes = padSuffixes;
        this.singlePass = singlePass;
    }

    /**
     * @param reuseSystemPrefix keep the schema catalog's KV cells from the previous call and prefill
     *     only the context (production-style warm cache). Cold calls prefill everything.
     */
    public Result run(Preset preset, boolean reuseSystemPrefix) {
        long t0 = System.nanoTime();
        int m = schema.fields.size();

        // --- tokenize ---------------------------------------------------------------------------
        int[] contextTokens = rt.tokenize(schema.userTurn(preset) + schema.userSuffix + "{\n", true);
        int sysLen = schema.systemTokens.length;
        int prefixLen = sysLen + contextTokens.length;
        long t1 = System.nanoTime();

        // --- prefill: sequence 0 only, or (single pass) tagged with every field's sequence -------
        int[] allSeqs = new int[m + 1];
        for (int s = 0; s <= m; s++) {
            allSeqs[s] = s;
        }
        List<LlamaRuntime.Tok> batch = new ArrayList<>();
        boolean warm = reuseSystemPrefix && cachedSystemTokens == sysLen;
        if (warm) {
            for (int s = 1; s <= m; s++) {
                rt.seqRemove(s, 0, -1);
            }
            rt.seqRemove(0, sysLen, -1);
        } else {
            rt.clearMemory();
            for (int i = 0; i < sysLen; i++) {
                batch.add(singlePass
                        ? new LlamaRuntime.Tok(schema.systemTokens[i], allSeqs, i, false)
                        : new LlamaRuntime.Tok(schema.systemTokens[i], 0, i, false));
            }
        }
        for (int i = 0; i < contextTokens.length; i++) {
            batch.add(singlePass
                    ? new LlamaRuntime.Tok(contextTokens[i], allSeqs, sysLen + i, false)
                    : new LlamaRuntime.Tok(contextTokens[i], 0, sysLen + i, false));
        }
        long t2;
        long t3;
        if (singlePass) {
            if (warm) {
                for (int s = 1; s <= m; s++) {
                    rt.seqCopy(0, s, 0, sysLen); // share the cached catalog cells (zero copy)
                }
            }
            t2 = System.nanoTime();
            t3 = t2;
        } else {
            rt.decode(batch);
            rt.synchronize();
            t2 = System.nanoTime();
            // --- broadcast the prefix to one sequence per field ------------------------------
            for (int s = 1; s <= m; s++) {
                rt.seqCopy(0, s, 0, prefixLen);
            }
            t3 = System.nanoTime();
            batch.clear();
        }
        cachedSystemTokens = sysLen;

        // --- one batched pass over all suffixes -------------------------------------------------
        int maxSuffix = 0;
        for (CompiledSchema.Field f : schema.fields) {
            maxSuffix = Math.max(maxSuffix, f.suffixTokens().length);
        }
        int[] curPos = new int[m + 1];
        for (int fi = 0; fi < m; fi++) {
            int[] suffix = schema.fields.get(fi).suffixTokens();
            int seq = fi + 1;
            for (int i = 0; i < suffix.length; i++) {
                batch.add(new LlamaRuntime.Tok(suffix[i], seq, prefixLen + i, i == suffix.length - 1));
            }
            if (padSuffixes) {
                for (int i = suffix.length; i < maxSuffix; i++) {
                    batch.add(new LlamaRuntime.Tok(suffix[suffix.length - 1], seq, prefixLen + i, false));
                }
            }
            curPos[seq] = prefixLen + suffix.length;
        }
        List<Integer> outputs = rt.decode(batch);
        rt.synchronize(); // llama_decode returns before the GPU finishes; sync so the breakdown is honest
        long t4 = System.nanoTime();

        // --- level 0 scoring, then batched token-tree levels for ambiguous fields ---------------
        FieldState[] states = new FieldState[m];
        List<Integer> pending = new ArrayList<>();
        for (int fi = 0; fi < m; fi++) {
            states[fi] = new FieldState(schema.fields.get(fi));
            if (!states[fi].advance(rt.logits(outputs.get(fi)))) {
                pending.add(fi);
            }
        }
        int levels = 0;
        while (!pending.isEmpty() && levels < MAX_LEVELS) {
            levels++;
            batch.clear();
            for (int fi : pending) {
                int seq = fi + 1;
                if (padSuffixes && levels == 1) {
                    rt.seqRemove(seq, curPos[seq], -1); // drop the pad cells before continuing
                }
                batch.add(new LlamaRuntime.Tok(states[fi].lastToken, seq, curPos[seq]++, true));
            }
            List<Integer> outs = rt.decode(batch);
            List<Integer> still = new ArrayList<>();
            for (int i = 0; i < pending.size(); i++) {
                int fi = pending.get(i);
                if (!states[fi].advance(rt.logits(outs.get(i)))) {
                    still.add(fi);
                }
            }
            pending = still;
        }
        for (int fi : pending) {
            states[fi].forceResolve();
        }
        List<FieldResult> results = new ArrayList<>();
        for (FieldState s : states) {
            results.add(s.toResult());
        }
        long t5 = System.nanoTime(); // same boundary as Python: assembly is inside the timer

        double ms = 1_000_000.0;
        return new Result((t5 - t0) / ms, (t1 - t0) / ms, (t2 - t1) / ms, (t3 - t2) / ms, (t4 - t3) / ms,
                (t5 - t4) / ms, prefixLen, levels, results);
    }

    /** Token-tree state of one field: which choices are still live and how far each has been matched. */
    private final class FieldState {
        final CompiledSchema.Field field;
        final int n;
        final boolean[] live;
        final int[] matched;      // tokens of each choice consumed so far
        final double[] prob;      // cumulative probability mass assigned to each choice
        int levels;
        int lastToken = -1;
        int winner = -1;

        FieldState(CompiledSchema.Field field) {
            this.field = field;
            this.n = field.choiceTokens().size();
            this.live = new boolean[n];
            this.matched = new int[n];
            this.prob = new double[n];
            java.util.Arrays.fill(live, true);
            java.util.Arrays.fill(prob, 1.0);
        }

        /** Next token each live choice needs; -1 encodes "any closing quote token". */
        private int nextToken(int c) {
            int[] toks = field.choiceTokens().get(c);
            return matched[c] < toks.length ? toks[matched[c]] : -1;
        }

        private double logitFor(MemorySegment logits, int token) {
            if (token >= 0) {
                return rt.logit(logits, token);
            }
            double best = Double.NEGATIVE_INFINITY;
            for (int q : schema.closingTokens) {
                best = Math.max(best, rt.logit(logits, q));
            }
            return best;
        }

        /**
         * Consumes the logits at the current position: softmax over the live choices' next tokens,
         * keep the choices sharing the winning token. Returns true once a single choice remains.
         */
        boolean advance(MemorySegment logits) {
            levels++;
            Map<Integer, Double> tokenScore = new LinkedHashMap<>();
            for (int c = 0; c < n; c++) {
                if (live[c]) {
                    tokenScore.computeIfAbsent(nextToken(c), t -> logitFor(logits, t));
                }
            }
            double max = tokenScore.values().stream().mapToDouble(Double::doubleValue).max().orElse(0);
            double denom = 0;
            for (double v : tokenScore.values()) {
                denom += Math.exp(v - max);
            }
            int bestToken = 0;
            double bestScore = Double.NEGATIVE_INFINITY;
            for (var e : tokenScore.entrySet()) {
                if (e.getValue() > bestScore) {
                    bestScore = e.getValue();
                    bestToken = e.getKey();
                }
            }
            // How many live choices share each next token: an eliminated group's mass is split evenly
            // among its members so the reported probabilities still form a distribution.
            Map<Integer, Integer> groupSize = new LinkedHashMap<>();
            for (int c = 0; c < n; c++) {
                if (live[c]) {
                    groupSize.merge(nextToken(c), 1, Integer::sum);
                }
            }
            int remaining = 0;
            int last = -1;
            for (int c = 0; c < n; c++) {
                if (!live[c]) {
                    continue;
                }
                int t = nextToken(c);
                prob[c] *= Math.exp(tokenScore.get(t) - max) / denom;
                if (t == bestToken) {
                    matched[c]++;
                    remaining++;
                    last = c;
                } else {
                    live[c] = false;
                    prob[c] /= groupSize.get(t);
                }
            }
            if (remaining == 1) {
                winner = last;
                return true;
            }
            // Several choices share this token: emit it and look at the next position.
            lastToken = bestToken >= 0 ? bestToken : schema.closingTokens[0];
            return false;
        }

        void forceResolve() {
            double best = -1;
            for (int c = 0; c < n; c++) {
                if (live[c] && prob[c] > best) {
                    best = prob[c];
                    winner = c;
                }
            }
        }

        FieldResult toResult() {
            String choice = field.def().choices().get(winner);
            Object value = field.def().isBoolean() ? Boolean.valueOf("true".equals(choice)) : choice;
            Map<String, Double> probs = new LinkedHashMap<>();
            for (int c = 0; c < n; c++) {
                probs.put(field.def().choices().get(c), prob[c]);
            }
            return new FieldResult(field.def().name(), value, prob[winner], levels, probs);
        }
    }
}
