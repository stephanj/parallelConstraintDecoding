package pcd;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;

/**
 * The GGUF models the engine can load: files in the project's {@code models/} directory plus the
 * models of a local Ollama install, whose store is a directory of GGUF blobs addressed by manifests.
 * Ollama models are identified as {@code ollama:<name>:<tag>}; nothing here talks to Ollama itself.
 */
public final class ModelCatalog {

    /** @param id stable identifier ({@code file name} or {@code ollama:name:tag}) */
    public record Entry(String id, String label, String source, Path path, long sizeBytes, String architecture) {}

    /** GGUF architectures that are not decoder LLMs (embedding models, vision projectors). */
    private static final Set<String> NOT_GENERATIVE = Set.of("bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "clip", "t5encoder");
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final Path modelsDir;
    private final Path ollamaDir;

    public ModelCatalog(Path modelsDir) {
        this(modelsDir, defaultOllamaDir());
    }

    public ModelCatalog(Path modelsDir, Path ollamaDir) {
        this.modelsDir = modelsDir;
        this.ollamaDir = ollamaDir;
    }

    static Path defaultOllamaDir() {
        String env = System.getenv("OLLAMA_MODELS");
        return env != null ? Path.of(env) : Path.of(System.getProperty("user.home"), ".ollama", "models");
    }

    public List<Entry> list() throws IOException {
        List<Entry> out = new ArrayList<>();
        if (Files.isDirectory(modelsDir)) {
            try (Stream<Path> files = Files.list(modelsDir)) {
                for (Path f : files.filter(f -> f.toString().endsWith(".gguf")).sorted().toList()) {
                    String arch = architecture(f);
                    if (arch == null || NOT_GENERATIVE.contains(arch)) {
                        continue;
                    }
                    String name = f.getFileName().toString();
                    out.add(new Entry(name, name.replaceFirst("\\.gguf$", ""), "local", f, Files.size(f), arch));
                }
            }
        }
        out.addAll(ollamaModels());
        return out;
    }

    public Entry find(String id) throws IOException {
        return list().stream().filter(e -> e.id().equals(id)).findFirst()
                .orElseThrow(() -> new IllegalArgumentException("unknown model: " + id));
    }

    /** Walks {@code manifests/<registry>/<namespace>/<name>/<tag>} and resolves each model layer blob. */
    private List<Entry> ollamaModels() {
        List<Entry> out = new ArrayList<>();
        Path manifests = ollamaDir.resolve("manifests");
        if (!Files.isDirectory(manifests)) {
            return out;
        }
        try (Stream<Path> walk = Files.walk(manifests)) {
            for (Path mf : walk.filter(Files::isRegularFile).sorted().toList()) {
                if (mf.getFileName().toString().startsWith("._")) {
                    continue; // macOS resource forks
                }
                Path rel = manifests.relativize(mf);
                if (rel.getNameCount() < 4) {
                    continue;
                }
                String namespace = rel.getName(rel.getNameCount() - 3).toString();
                String name = rel.getName(rel.getNameCount() - 2).toString() + ":" + rel.getFileName();
                if (!namespace.equals("library")) {
                    name = namespace + "/" + name;
                }
                try {
                    JsonNode manifest = MAPPER.readTree(Files.readString(mf));
                    for (JsonNode layer : manifest.path("layers")) {
                        if (!layer.path("mediaType").asText("").endsWith("image.model")) {
                            continue;
                        }
                        Path blob = ollamaDir.resolve("blobs").resolve(layer.path("digest").asText().replace(':', '-'));
                        if (!Files.isRegularFile(blob)) {
                            continue;
                        }
                        String arch = architecture(blob);
                        if (arch == null || NOT_GENERATIVE.contains(arch)) {
                            continue;
                        }
                        out.add(new Entry("ollama:" + name, "ollama:" + name, "ollama", blob, Files.size(blob), arch));
                    }
                } catch (IOException | RuntimeException e) {
                    System.err.println("skipping Ollama manifest " + mf + ": " + e.getMessage());
                }
            }
        } catch (IOException e) {
            System.err.println("cannot read Ollama store " + ollamaDir + ": " + e.getMessage());
        }
        out.sort(Comparator.comparing(Entry::label));
        return out;
    }

    /**
     * Reads {@code general.architecture} from a GGUF header without loading the model. Returns null
     * for files that are not GGUF or whose header does not carry the key early on.
     */
    static String architecture(Path file) {
        try (RandomAccessFile raf = new RandomAccessFile(file.toFile(), "r")) {
            byte[] head = new byte[Math.min((int) Math.min(raf.length(), 1 << 20), 1 << 20)];
            raf.readFully(head);
            ByteBuffer b = ByteBuffer.wrap(head).order(ByteOrder.LITTLE_ENDIAN);
            if (b.getInt() != 0x46554747) { // "GGUF"
                return null;
            }
            int version = b.getInt();
            if (version < 2) {
                return null;
            }
            b.getLong(); // tensor count
            long nKv = b.getLong();
            for (long i = 0; i < nKv && i < 64; i++) {
                String key = string(b);
                int type = b.getInt();
                if (key.equals("general.architecture") && type == 8) {
                    return string(b);
                }
                skip(b, type);
            }
            return null;
        } catch (IOException | RuntimeException e) {
            return null;
        }
    }

    private static String string(ByteBuffer b) {
        long len = b.getLong();
        if (len < 0 || len > b.remaining()) {
            throw new IllegalStateException("bad GGUF string length " + len);
        }
        byte[] bytes = new byte[(int) len];
        b.get(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private static void skip(ByteBuffer b, int type) {
        switch (type) {
            case 0, 1, 7 -> b.get();
            case 2, 3 -> b.getShort();
            case 4, 5, 6 -> b.getInt();
            case 8 -> string(b);
            case 9 -> {
                int elem = b.getInt();
                long n = b.getLong();
                for (long i = 0; i < n; i++) {
                    skip(b, elem);
                }
            }
            case 10, 11, 12 -> b.getLong();
            default -> throw new IllegalStateException("unknown GGUF value type " + type);
        }
    }
}
