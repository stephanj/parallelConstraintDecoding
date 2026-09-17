package pcd;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Refreshes the Devoxx CFP routing preset from the public CFP API: the real track list becomes
 * the {@code track} choices, and the N most-favourited talks become the sample bank, each with
 * the track / format / level the CFP actually filed it under.
 *
 * <pre>
 *   java -jar target/pcd-benchmark.jar devoxx-samples [--event dvbe26] [--count 100]
 * </pre>
 */
public final class DevoxxSamples {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    /** CFP session types that map onto the preset's session_format choices. */
    private static final Set<String> FORMATS = Set.of("Conference", "Deep Dive", "Tools-in-Action", "Hands-on Lab", "Lunch Talk", "BOF");

    public static void main(String[] args) throws Exception {
        String event = "dvbe26";
        int count = 100;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--event" -> event = args[++i];
                case "--count" -> count = Integer.parseInt(args[++i]);
                default -> throw new IllegalArgumentException("unknown option " + args[i]);
            }
        }
        Path file = Main.repoRoot().resolve("presets").resolve("devoxx_cfp.json");
        Preset preset = Preset.load(file);

        HttpClient http = HttpClient.newHttpClient();
        JsonNode tracks = get(http, "https://" + event + ".cfp.dev/api/public/tracks");
        JsonNode talks = get(http, "https://" + event + ".cfp.dev/api/public/talks");

        List<String> trackNames = new ArrayList<>();
        tracks.forEach(t -> trackNames.add(t.path("name").asText()));
        Map<String, Preset.FieldDef> schema = new LinkedHashMap<>(preset.schema());
        Preset.FieldDef track = schema.get("track");
        schema.put("track", new Preset.FieldDef("track", "enum", track.description(), List.copyOf(trackNames)));

        List<JsonNode> eligible = new ArrayList<>();
        talks.forEach(t -> {
            boolean ok = FORMATS.contains(t.path("sessionType").path("name").asText())
                    && stripHtml(t.path("description").asText("")).length() >= 200
                    && trackNames.contains(t.path("track").path("name").asText());
            if (ok) {
                eligible.add(t);
            }
        });
        eligible.sort(Comparator.comparingInt((JsonNode t) -> t.path("totalFavourites").asInt()).reversed());

        List<Preset.Sample> samples = new ArrayList<>();
        for (JsonNode t : eligible.subList(0, Math.min(count, eligible.size()))) {
            String speakers = new ArrayList<>(t.path("speakers").findValuesAsText("fullName")).stream()
                    .collect(Collectors.joining(", "));
            String context = "CFP SUBMISSION #" + t.path("id").asText() + " - " + eventLabel(event) + "\n"
                    + "Title: " + t.path("title").asText() + "\n"
                    + "Speakers: " + speakers + "\n\n"
                    + "Abstract:\n" + stripHtml(t.path("description").asText());
            Map<String, String> expected = new LinkedHashMap<>();
            expected.put("track", t.path("track").path("name").asText());
            expected.put("session_format", t.path("sessionType").path("name").asText());
            expected.put("audience_level", t.path("audienceLevel").asText());
            samples.add(new Preset.Sample("#" + t.path("id").asText() + " " + t.path("title").asText(), context, expected));
        }
        if (samples.isEmpty()) {
            throw new IllegalStateException("no eligible talks found for " + event);
        }

        Preset updated = new Preset(preset.id(), preset.title(), preset.description(),
                samples.get(0).context(), schema, samples);
        updated.save(file);
        System.out.printf("%s: %d tracks, %d talks cached (of %d eligible) -> %s%n",
                event, trackNames.size(), samples.size(), eligible.size(), file);
    }

    private static JsonNode get(HttpClient http, String url) throws Exception {
        HttpResponse<String> r = http.send(HttpRequest.newBuilder(URI.create(url)).GET().build(), HttpResponse.BodyHandlers.ofString());
        if (r.statusCode() != 200) {
            throw new IllegalStateException(url + " -> HTTP " + r.statusCode());
        }
        return MAPPER.readTree(r.body());
    }

    private static String eventLabel(String slug) {
        return slug.startsWith("dvbe") ? "Devoxx Belgium 20" + slug.substring(4) : slug;
    }

    /** Turns the CFP's HTML abstract into readable plain text with list items and paragraphs kept. */
    static String stripHtml(String html) {
        String s = html.replaceAll("(?i)</(p|li|ul|ol|div|h\\d)>", "\n")
                .replaceAll("(?i)<br\\s*/?>", "\n")
                .replaceAll("(?i)<li[^>]*>", "- ")
                .replaceAll("<[^>]+>", "");
        s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                .replace("&quot;", "\"").replace("&#39;", "'").replace("&rsquo;", "’").replace("&ldquo;", "“")
                .replace("&rdquo;", "”").replace("&hellip;", "…").replace("&ndash;", "–").replace("&mdash;", "—");
        return s.replaceAll("\n{3,}", "\n\n").strip();
    }
}
