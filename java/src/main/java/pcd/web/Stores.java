package pcd.web;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;
import pcd.Preset;

/** File-backed storage: presets in {@code presets/*.json}, benchmark runs in {@code results/*.json}. */
final class Stores {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    /** Presets directory — the same files the CLI benchmark reads. */
    static final class Presets {
        private final Path dir;

        Presets(Path dir) throws IOException {
            this.dir = dir;
            Files.createDirectories(dir);
        }

        List<Preset> list() throws IOException {
            try (Stream<Path> files = Files.list(dir)) {
                List<Preset> out = new ArrayList<>();
                for (Path p : files.filter(f -> f.toString().endsWith(".json")).sorted().toList()) {
                    try {
                        out.add(Preset.load(p));
                    } catch (IllegalArgumentException e) {
                        System.err.println("skipping " + p + ": " + e.getMessage());
                    }
                }
                return out;
            }
        }

        Preset get(String id) throws IOException {
            Path p = path(id);
            if (!Files.exists(p)) {
                throw new NotFound("preset '" + id + "' not found");
            }
            return Preset.load(p);
        }

        boolean exists(String id) {
            return Files.exists(path(id));
        }

        void save(Preset preset) throws IOException {
            preset.save(path(preset.id()));
        }

        boolean delete(String id) throws IOException {
            return Files.deleteIfExists(path(id));
        }

        private Path path(String id) {
            if (!id.matches("[a-z0-9_]{1,64}")) {
                throw new IllegalArgumentException("invalid preset id");
            }
            return dir.resolve(id + ".json");
        }
    }

    /** One JSON document per benchmark run, newest first on read. */
    static final class Runs {
        private final Path dir;

        Runs(Path dir) throws IOException {
            this.dir = dir;
            Files.createDirectories(dir);
        }

        ObjectNode save(ObjectNode run) throws IOException {
            String stamp = DateTimeFormatter.ISO_INSTANT.format(Instant.now()).replace(":", "-");
            run.put("id", stamp);
            Files.writeString(dir.resolve(stamp + ".json"), MAPPER.writerWithDefaultPrettyPrinter().writeValueAsString(run));
            return run;
        }

        ArrayNode list(int limit) throws IOException {
            ArrayNode out = MAPPER.createArrayNode();
            try (Stream<Path> files = Files.list(dir)) {
                for (Path p : files.filter(f -> f.toString().endsWith(".json"))
                        .sorted(Comparator.reverseOrder()).limit(limit).toList()) {
                    JsonNode n = MAPPER.readTree(Files.readString(p));
                    out.add(n);
                }
            }
            return out;
        }
    }

    static final class NotFound extends RuntimeException {
        NotFound(String msg) {
            super(msg);
        }
    }
}
