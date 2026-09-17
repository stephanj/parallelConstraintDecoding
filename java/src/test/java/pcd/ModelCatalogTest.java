package pcd;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ModelCatalogTest {

    /** A minimal GGUF v3 header: magic, version, 0 tensors, the given string KV pairs. */
    private static byte[] gguf(String... kv) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ByteBuffer b = ByteBuffer.allocate(4096).order(ByteOrder.LITTLE_ENDIAN);
        b.put("GGUF".getBytes(StandardCharsets.US_ASCII)).putInt(3).putLong(0).putLong(kv.length / 2);
        for (int i = 0; i < kv.length; i += 2) {
            putString(b, kv[i]);
            b.putInt(8);
            putString(b, kv[i + 1]);
        }
        out.write(b.array(), 0, b.position());
        return out.toByteArray();
    }

    private static void putString(ByteBuffer b, String s) {
        byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
        b.putLong(bytes.length).put(bytes);
    }

    @Test
    void readsArchitectureFromHeader(@TempDir Path dir) throws IOException {
        Path f = dir.resolve("m.gguf");
        Files.write(f, gguf("general.name", "x", "general.architecture", "qwen2"));
        assertEquals("qwen2", ModelCatalog.architecture(f));
    }

    @Test
    void nonGgufFilesAreNull(@TempDir Path dir) throws IOException {
        Path f = dir.resolve("m.gguf");
        Files.writeString(f, "not a model");
        assertNull(ModelCatalog.architecture(f));
    }

    @Test
    void listsLocalModelsAndSkipsEmbeddingModels(@TempDir Path dir) throws IOException {
        Files.write(dir.resolve("chat.gguf"), gguf("general.architecture", "llama"));
        Files.write(dir.resolve("embed.gguf"), gguf("general.architecture", "bert"));
        Files.writeString(dir.resolve("notes.txt"), "ignored");

        List<ModelCatalog.Entry> entries = new ModelCatalog(dir, dir.resolve("no-ollama-here")).list();

        assertEquals(1, entries.size());
        assertEquals("chat.gguf", entries.get(0).id());
        assertEquals("local", entries.get(0).source());
        assertEquals("llama", entries.get(0).architecture());
    }

    @Test
    void resolvesOllamaManifestsToBlobsWithPrefixedIds(@TempDir Path dir) throws IOException {
        Path ollama = dir.resolve("ollama");
        Path blobs = Files.createDirectories(ollama.resolve("blobs"));
        Files.write(blobs.resolve("sha256-abc"), gguf("general.architecture", "llama"));
        Path manifest = Files.createDirectories(ollama.resolve("manifests/registry.ollama.ai/library/llama3.1")).resolve("latest");
        Files.writeString(manifest, """
            {"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:abc","size":1},
                       {"mediaType":"application/vnd.ollama.image.template","digest":"sha256:zzz","size":1}]}""");

        List<ModelCatalog.Entry> entries = new ModelCatalog(dir.resolve("empty"), ollama).list();

        assertEquals(1, entries.size());
        assertEquals("ollama:llama3.1:latest", entries.get(0).id());
        assertEquals("ollama", entries.get(0).source());
        assertEquals(blobs.resolve("sha256-abc"), entries.get(0).path());
    }
}
