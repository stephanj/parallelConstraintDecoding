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
 *   java -jar target/pcd-benchmark.jar serve [--port 8000]      # web UI
 *   java -jar target/pcd-benchmark.jar bench [flags] [preset...] # CLI benchmark (default)
 * </pre>
 */
public final class Main {

    public static void main(String[] args) throws Exception {
        if (args.length > 0 && args[0].equals("serve")) {
            int port = 8000;
            for (int i = 1; i < args.length; i++) {
                if (args[i].equals("--port")) {
                    port = Integer.parseInt(args[++i]);
                }
            }
            Path root = repoRoot();
            Server server = new Server(port, NativeBenchmark.resolveModel(), root.resolve("presets"), root.resolve("results"));
            Runtime.getRuntime().addShutdownHook(new Thread(server::stop));
            server.start();
            Thread.currentThread().join();
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
