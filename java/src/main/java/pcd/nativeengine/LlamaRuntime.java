package pcd.nativeengine;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import llama.Llama;
import llama.llama_batch;
import llama.llama_context_params;
import llama.llama_model_params;

/**
 * Thin FFM wrapper over libllama: model/context lifecycle, tokenization, batched decode, logits and
 * the KV-memory sequence operations the parallel engine needs. All native memory lives in one arena
 * owned by this object.
 */
public final class LlamaRuntime implements AutoCloseable {

    /** Homebrew installs ggml and llama.cpp as separate formulas; both lib dirs are searched. */
    static final List<String> DEFAULT_LIB_DIRS = List.of("/opt/homebrew/opt/ggml/lib", "/opt/homebrew/opt/llama.cpp/lib");

    private static boolean libsLoaded;

    private final Arena arena = Arena.ofShared();
    private final MemorySegment model;
    private final MemorySegment ctx;
    private final MemorySegment vocab;
    private final MemorySegment memory;
    private final int nVocab;
    private final MemorySegment batch;
    private final int batchCapacity;
    private final MemorySegment tokenScratch;
    private final MemorySegment pieceScratch;
    private int maxSeqPerToken;

    public record Options(Path modelPath, int nCtx, int nSeqMax, int nBatch, boolean flashAttention, boolean kvUnified) {
        public static Options defaults(Path modelPath) {
            return new Options(modelPath, 8192, 64, 2048, true, true);
        }
    }

    public LlamaRuntime(Options opts) {
        loadLibraries();
        Llama.llama_backend_init();
        Llama.ggml_backend_load_all();
        installQuietLogger();

        MemorySegment mparams = Llama.llama_model_default_params(arena);
        llama_model_params.n_gpu_layers(mparams, 99);
        model = Llama.llama_model_load_from_file(arena.allocateFrom(opts.modelPath().toString()), mparams);
        if (model.equals(MemorySegment.NULL)) {
            throw new IllegalStateException("Failed to load model " + opts.modelPath());
        }

        MemorySegment cparams = Llama.llama_context_default_params(arena);
        llama_context_params.n_ctx(cparams, opts.nCtx());
        llama_context_params.n_batch(cparams, opts.nBatch());
        llama_context_params.n_ubatch(cparams, opts.nBatch());
        llama_context_params.n_seq_max(cparams, opts.nSeqMax());
        llama_context_params.flash_attn_type(cparams,
                opts.flashAttention() ? Llama.LLAMA_FLASH_ATTN_TYPE_ENABLED() : Llama.LLAMA_FLASH_ATTN_TYPE_DISABLED());
        llama_context_params.kv_unified(cparams, opts.kvUnified());
        llama_context_params.no_perf(cparams, true);
        ctx = Llama.llama_init_from_model(model, cparams);
        if (ctx.equals(MemorySegment.NULL)) {
            throw new IllegalStateException("Failed to create llama context");
        }

        vocab = Llama.llama_model_get_vocab(model);
        memory = Llama.llama_get_memory(ctx);
        nVocab = Llama.llama_vocab_n_tokens(vocab);
        batchCapacity = opts.nBatch();
        batch = Llama.llama_batch_init(arena, batchCapacity, 0, opts.nSeqMax());
        maxSeqPerToken = opts.nSeqMax();
        tokenScratch = arena.allocate(ValueLayout.JAVA_INT, opts.nCtx());
        pieceScratch = arena.allocate(256);
    }

    private static synchronized void loadLibraries() {
        if (libsLoaded) {
            return;
        }
        String override = System.getenv("PCD_LLAMA_LIB_DIR");
        List<String> dirs = override == null ? DEFAULT_LIB_DIRS : List.of(override.split(":"));
        for (String lib : List.of("libggml-base.dylib", "libggml.dylib", "libllama.dylib")) {
            Path found = dirs.stream().map(d -> Path.of(d, lib)).filter(Files::exists).findFirst()
                    .orElseThrow(() -> new IllegalStateException(
                            "Missing " + lib + " in " + dirs + " (brew install llama.cpp, or set PCD_LLAMA_LIB_DIR=dir[:dir])"));
            System.load(found.toString());
        }
        libsLoaded = true;
    }

    /** Routes llama.cpp logging through an upcall that only lets errors through. */
    private void installQuietLogger() {
        try {
            var handle = MethodHandles.lookup().findStatic(LlamaRuntime.class, "onLog",
                    java.lang.invoke.MethodType.methodType(void.class, int.class, MemorySegment.class, MemorySegment.class));
            var descriptor = FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS);
            MemorySegment stub = Linker.nativeLinker().upcallStub(handle, descriptor, arena);
            Llama.llama_log_set(stub, MemorySegment.NULL);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }

    private static void onLog(int level, MemorySegment text, MemorySegment userData) {
        if (level >= 4) { // GGML_LOG_LEVEL_ERROR
            System.err.print(text.reinterpret(Long.MAX_VALUE).getString(0));
        }
    }

    public int vocabSize() {
        return nVocab;
    }

    public boolean isEndOfGeneration(int token) {
        return Llama.llama_vocab_is_eog(vocab, token);
    }

    /** Copies the logits of batch entry {@code batchIndex} into a heap array and returns the argmax. */
    public int argmax(int batchIndex, float[] scratch) {
        MemorySegment seg = logits(batchIndex);
        MemorySegment.copy(seg, ValueLayout.JAVA_FLOAT, 0, scratch, 0, nVocab);
        int best = 0;
        for (int i = 1; i < nVocab; i++) {
            if (scratch[i] > scratch[best]) {
                best = i;
            }
        }
        return best;
    }

    /** Tokenizes text; special tokens such as {@code <|im_start|>} are parsed when parseSpecial is set. */
    public int[] tokenize(String text, boolean parseSpecial) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        MemorySegment buf = arena.allocate(bytes.length + 1);
        MemorySegment.copy(bytes, 0, buf, ValueLayout.JAVA_BYTE, 0, bytes.length);
        int n = Llama.llama_tokenize(vocab, buf, bytes.length, tokenScratch, (int) tokenScratch.byteSize() / 4, false, parseSpecial);
        if (n < 0) {
            throw new IllegalStateException("Prompt of " + (-n) + " tokens exceeds context " + tokenScratch.byteSize() / 4);
        }
        int[] out = new int[n];
        MemorySegment.copy(tokenScratch, ValueLayout.JAVA_INT, 0, out, 0, n);
        return out;
    }

    public String piece(int token) {
        int n = Llama.llama_token_to_piece(vocab, token, pieceScratch, (int) pieceScratch.byteSize(), 0, false);
        if (n < 0) {
            return "";
        }
        byte[] bytes = new byte[n];
        MemorySegment.copy(pieceScratch, ValueLayout.JAVA_BYTE, 0, bytes, 0, n);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    /**
     * One token to place in a batch: the sequences it belongs to, its position, and whether logits
     * are wanted for it. A token tagged with several sequences lands in the KV cache once and is
     * visible to all of them — that is how the shared prefix is broadcast inside a single decode.
     */
    public record Tok(int token, int[] seqs, int pos, boolean logits) {
        public Tok(int token, int seq, int pos, boolean logits) {
            this(token, new int[] {seq}, pos, logits);
        }
    }

    /**
     * Runs one llama_decode over the given tokens. Returns, for each entry with logits requested,
     * its index in the batch (to pass to {@link #logits(int)}), in order.
     */
    public List<Integer> decode(List<Tok> toks) {
        if (toks.size() > batchCapacity) {
            throw new IllegalArgumentException("batch of " + toks.size() + " exceeds n_batch " + batchCapacity);
        }
        MemorySegment token = llama_batch.token(batch);
        MemorySegment pos = llama_batch.pos(batch);
        MemorySegment nSeqId = llama_batch.n_seq_id(batch);
        MemorySegment seqId = llama_batch.seq_id(batch);
        MemorySegment logits = llama_batch.logits(batch);
        List<Integer> outputs = new ArrayList<>();
        for (int i = 0; i < toks.size(); i++) {
            Tok t = toks.get(i);
            token.setAtIndex(ValueLayout.JAVA_INT, i, t.token());
            pos.setAtIndex(ValueLayout.JAVA_INT, i, t.pos());
            if (t.seqs().length > maxSeqPerToken) {
                throw new IllegalArgumentException("token tagged with more sequences than n_seq_max");
            }
            nSeqId.setAtIndex(ValueLayout.JAVA_INT, i, t.seqs().length);
            MemorySegment seqPtr = seqId.getAtIndex(ValueLayout.ADDRESS, i).reinterpret(4L * t.seqs().length);
            for (int k = 0; k < t.seqs().length; k++) {
                seqPtr.setAtIndex(ValueLayout.JAVA_INT, k, t.seqs()[k]);
            }
            logits.set(ValueLayout.JAVA_BYTE, i, (byte) (t.logits() ? 1 : 0));
            if (t.logits()) {
                outputs.add(i);
            }
        }
        llama_batch.n_tokens(batch, toks.size());
        int rc = Llama.llama_decode(ctx, batch);
        if (rc != 0) {
            throw new IllegalStateException("llama_decode failed with " + rc);
        }
        return outputs;
    }

    /** Logits for batch entry {@code batchIndex} (must have had logits requested) as a live view. */
    public MemorySegment logits(int batchIndex) {
        return Llama.llama_get_logits_ith(ctx, batchIndex).reinterpret((long) nVocab * Float.BYTES);
    }

    public float logit(MemorySegment logits, int token) {
        return logits.getAtIndex(ValueLayout.JAVA_FLOAT, token);
    }

    public void clearMemory() {
        Llama.llama_memory_clear(memory, true);
    }

    /** Tags the cells of {@code src} in [p0, p1) with {@code dst} (zero-copy under kv_unified). */
    public void seqCopy(int src, int dst, int p0, int p1) {
        Llama.llama_memory_seq_cp(memory, src, dst, p0, p1);
    }

    public void seqRemove(int seq, int p0, int p1) {
        Llama.llama_memory_seq_rm(memory, seq, p0, p1);
    }

    public void synchronize() {
        Llama.llama_synchronize(ctx);
    }

    @Override
    public void close() {
        Llama.llama_batch_free(batch);
        Llama.llama_free(ctx);
        Llama.llama_model_free(model);
        Llama.llama_backend_free();
        arena.close();
    }
}
