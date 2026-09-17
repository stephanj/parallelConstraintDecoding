package pcd;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import pcd.web.Server;

/**
 * Entry point of the jar.
 *
 * <pre>
 *   java -jar target/pcd-benchmark.jar serve [--port 8000] [--bind 0.0.0.0] [--read-only]   # web UI
 *   java -jar target/pcd-benchmark.jar bench [flags] [preset...] # CLI benchmark (default)
 *   java -jar target/pcd-benchmark.jar devoxx-samples [--event dvbe26] [--count 100]
 * </pre>
 */
public final class Main {

    public static void main(String[] args) throws Exception {
        if (args.length > 0 && args[0].equals("serve")) {
            // Defaults suit a laptop (loopback:8000); PORT / PCD_BIND / PCD_READ_ONLY suit a container host.
            int port = Integer.parseInt(System.getenv().getOrDefault("PORT", "8000"));
            String bind = System.getenv().getOrDefault("PCD_BIND", "127.0.0.1");
            boolean readOnly = Boolean.parseBoolean(System.getenv().getOrDefault("PCD_READ_ONLY", "false"));
            for (int i = 1; i < args.length; i++) {
                switch (args[i]) {
                    case "--port" -> port = Integer.parseInt(args[++i]);
                    case "--bind" -> bind = args[++i];
                    case "--read-only" -> readOnly = true;
                    default -> throw new IllegalArgumentException("unknown option " + args[i]);
                }
            }
            Path root = repoRoot();
            Server server = new Server(bind, port, readOnly, NativeBenchmark.resolveModel(),
                    root.resolve("presets"), root.resolve("results"));
            Runtime.getRuntime().addShutdownHook(new Thread(server::stop));
            server.start();
            Thread.currentThread().join();
            return;
        }
        if (args.length > 0 && args[0].equals("devoxx-samples")) {
            DevoxxSamples.main(java.util.Arrays.copyOfRange(args, 1, args.length));
            return;
        }
        List<String> rest = new ArrayList<>(List.of(args));
        if (!rest.isEmpty() && rest.get(0).equals("bench")) {
            rest.remove(0);
        }
        NativeBenchmark.main(rest.toArray(String[]::new));
    }

    /** The repo root is wherever presets/ lives: the current directory or its parent (java/). */
    static Path repoRoot() {
        return Files.isDirectory(Path.of("presets")) ? Path.of(".") : Path.of("..");
    }
}
