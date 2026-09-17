package pcd.web;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import pcd.Preset;

/**
 * Local web app: results dashboard, preset editor and the live "race" demo, served from one jar.
 *
 * <pre>
 *   java -jar target/pcd-benchmark.jar serve [--port 8000]
 * </pre>
 */
public final class Server {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final Map<String, String> MIME = Map.of(
            "html", "text/html; charset=utf-8",
            "js", "text/javascript; charset=utf-8",
            "css", "text/css; charset=utf-8",
            "svg", "image/svg+xml",
            "png", "image/png");

    private final EngineService engine;
    private final Stores.Presets presets;
    private final Stores.Runs runs;
    private final HttpServer http;

    public Server(int port, Path modelPath, Path presetsDir, Path resultsDir) throws IOException {
        this.presets = new Stores.Presets(presetsDir);
        this.runs = new Stores.Runs(resultsDir);
        System.out.println("Loading " + modelPath + " ...");
        long t0 = System.nanoTime();
        this.engine = new EngineService(modelPath);
        System.out.printf("Engine loaded in %.1fs%n", (System.nanoTime() - t0) / 1e9);

        http = HttpServer.create(new InetSocketAddress("127.0.0.1", port), 0);
        http.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        http.createContext("/api/status", ex -> handle(ex, this::status));
        http.createContext("/api/presets", ex -> handle(ex, this::presetsApi));
        http.createContext("/api/run/parallel", ex -> handle(ex, this::runParallel));
        http.createContext("/api/run/race", ex -> handle(ex, this::runRace));
        http.createContext("/api/benchmark", ex -> handle(ex, this::benchmark));
        http.createContext("/api/results", ex -> handle(ex, this::results));
        http.createContext("/", ex -> handle(ex, this::staticFile));
    }

    public void start() {
        http.start();
        System.out.println("Parallel Constrained Decoding UI: http://127.0.0.1:" + http.getAddress().getPort());
    }

    public void stop() {
        http.stop(0);
        engine.close();
    }

    // ---------------------------------------------------------------- handlers

    private void status(HttpExchange ex) throws IOException {
        ObjectNode out = MAPPER.createObjectNode();
        out.put("model", engine.modelPath.getFileName().toString());
        out.put("presets", presets.list().size());
        json(ex, 200, out);
    }

    private void presetsApi(HttpExchange ex) throws IOException {
        String path = ex.getRequestURI().getPath();
        String id = path.length() > "/api/presets/".length() ? path.substring("/api/presets/".length()) : null;
        switch (ex.getRequestMethod()) {
            case "GET" -> {
                if (id == null) {
                    ArrayNode arr = MAPPER.createArrayNode();
                    for (Preset p : presets.list()) {
                        ObjectNode n = arr.addObject();
                        n.put("id", p.id()).put("title", p.title()).put("description", p.description());
                        n.put("fields", p.schema().size());
                        n.put("contextChars", p.context().length());
                        n.put("samples", p.samples().size());
                    }
                    json(ex, 200, arr);
                } else {
                    json(ex, 200, presets.get(id).toJson());
                }
            }
            case "PUT" -> {
                ObjectNode body = (ObjectNode) MAPPER.readTree(ex.getRequestBody());
                if (id != null) {
                    body.put("id", id);
                }
                Preset p = Preset.fromJson(body);
                if (!body.has("samples") && id != null && presets.exists(id)) {
                    p = p.withSamples(presets.get(id).samples()); // the editor does not manage samples; keep them
                }
                presets.save(p);
                json(ex, 200, p.toJson());
            }
            case "DELETE" -> {
                if (id == null || !presets.delete(id)) {
                    throw new Stores.NotFound("preset not found");
                }
                json(ex, 200, MAPPER.createObjectNode().put("deleted", id));
            }
            default -> ex.sendResponseHeaders(405, -1);
        }
    }

    private void runParallel(HttpExchange ex) throws IOException {
        Preset p = presetFromBody(ex);
        json(ex, 200, engine.runParallel(p));
    }

    /**
     * SSE: the parallel result first (it is what would finish first), then the baseline's tokens as
     * they are generated, then the baseline's result. The UI replays both on one timeline.
     */
    private void runRace(HttpExchange ex) throws IOException {
        Preset p = presetFromBody(ex);
        try (Sse sse = Sse.open(ex)) {
            sse.send("parallel", engine.runParallel(p));
            sse.send("start", MAPPER.createObjectNode());
            ObjectNode naive = engine.runNaive(p, piece -> sse.send("token", MAPPER.createObjectNode().put("t", piece)));
            sse.send("naive", naive);
        }
    }

    /** SSE: runs every preset through both engines, emits one event per preset, stores the run. */
    private void benchmark(HttpExchange ex) throws IOException {
        List<Preset> all = presets.list();
        ObjectNode run = MAPPER.createObjectNode();
        run.put("timestamp", java.time.Instant.now().toString());
        run.put("model", engine.modelPath.getFileName().toString());
        ArrayNode rows = run.putArray("presets");
        try (Sse sse = Sse.open(ex)) {
            for (Preset p : all) {
                engine.runParallel(p); // warm-up for this schema's shapes
                ObjectNode parallel = engine.runParallel(p);
                ObjectNode naive = engine.runNaive(p, piece -> { });
                ObjectNode row = rows.addObject();
                row.put("id", p.id()).put("title", p.title()).put("fields", p.schema().size());
                row.set("parallel", parallel);
                naive.remove("text");
                naive.remove("json");
                row.set("naive", naive);
                row.put("speedup", Math.round(naive.get("elapsedMs").asDouble() / Math.max(parallel.get("elapsedMs").asDouble(), 1) * 10) / 10.0);
                sse.send("preset", row);
            }
            runs.save(run);
            sse.send("done", run);
        }
    }

    private void results(HttpExchange ex) throws IOException {
        json(ex, 200, runs.list(20));
    }

    private void staticFile(HttpExchange ex) throws IOException {
        String path = ex.getRequestURI().getPath();
        if (path.equals("/")) {
            path = "/index.html";
        }
        if (path.contains("..")) {
            ex.sendResponseHeaders(400, -1);
            return;
        }
        try (InputStream in = Server.class.getResourceAsStream("/web" + path)) {
            if (in == null) {
                ex.sendResponseHeaders(404, -1);
                return;
            }
            byte[] bytes = in.readAllBytes();
            String ext = path.substring(path.lastIndexOf('.') + 1);
            ex.getResponseHeaders().set("Content-Type", MIME.getOrDefault(ext, "application/octet-stream"));
            ex.getResponseHeaders().set("Cache-Control", "no-cache");
            ex.sendResponseHeaders(200, bytes.length);
            try (OutputStream out = ex.getResponseBody()) {
                out.write(bytes);
            }
        }
    }

    // ---------------------------------------------------------------- plumbing

    private interface Handler {
        void handle(HttpExchange ex) throws IOException;
    }

    private static void handle(HttpExchange ex, Handler h) throws IOException {
        try {
            h.handle(ex);
        } catch (Stores.NotFound e) {
            error(ex, 404, e.getMessage());
        } catch (IllegalArgumentException e) {
            error(ex, 400, e.getMessage());
        } catch (Exception e) {
            e.printStackTrace();
            error(ex, 500, e.toString());
        } finally {
            ex.close();
        }
    }

    /** A run request is either {@code {"presetId": ...}} or a full preset body (with an optional id). */
    private Preset presetFromBody(HttpExchange ex) throws IOException {
        ObjectNode body = (ObjectNode) MAPPER.readTree(ex.getRequestBody());
        if (body.hasNonNull("presetId") && !body.has("schema")) {
            return presets.get(body.get("presetId").asText());
        }
        if (!body.has("id")) {
            body.put("id", "adhoc");
        }
        if (!body.has("title")) {
            body.put("title", "Ad-hoc");
        }
        return Preset.fromJson(body);
    }

    private static void json(HttpExchange ex, int status, JsonNode body) throws IOException {
        byte[] bytes = MAPPER.writeValueAsBytes(body);
        ex.getResponseHeaders().set("Content-Type", "application/json");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = ex.getResponseBody()) {
            out.write(bytes);
        }
    }

    private static void error(HttpExchange ex, int status, String message) throws IOException {
        if (ex.getResponseHeaders().getFirst("Content-Type") != null) {
            return; // headers already sent (e.g. mid-stream); nothing sensible left to do
        }
        json(ex, status, MAPPER.createObjectNode().put("error", message == null ? "error" : message));
    }

    /** Minimal Server-Sent Events writer over the JDK HttpServer's chunked response. */
    private static final class Sse implements AutoCloseable {
        private final OutputStream out;

        private Sse(OutputStream out) {
            this.out = out;
        }

        static Sse open(HttpExchange ex) throws IOException {
            ex.getResponseHeaders().set("Content-Type", "text/event-stream");
            ex.getResponseHeaders().set("Cache-Control", "no-cache");
            ex.sendResponseHeaders(200, 0);
            return new Sse(ex.getResponseBody());
        }

        void send(String event, JsonNode data) {
            try {
                out.write(("event: " + event + "\ndata: " + MAPPER.writeValueAsString(data) + "\n\n")
                        .getBytes(StandardCharsets.UTF_8));
                out.flush();
            } catch (IOException e) {
                throw new IllegalStateException("client disconnected", e);
            }
        }

        @Override
        public void close() throws IOException {
            out.close();
        }
    }
}
