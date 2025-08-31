package com.sam.voice;

import org.vosk.LibVosk;
import org.vosk.Model;
import org.vosk.Recognizer;
import org.vosk.LogLevel;

import javax.sound.sampled.*;
import com.sun.speech.freetts.Voice;
import com.sun.speech.freetts.VoiceManager;

import java.io.*;
import java.net.URI;
import java.net.http.*;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.Locale;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * VoiceAssistant.java
 * Simple assistant using Vosk (offline STT) + FreeTTS (TTS).
 *
 * Run with: mvn exec:java -Dexec.mainClass="com.sam.voice.VoiceAssistant"
 *
 * Ensure MODEL_PATH points to an unzipped Vosk model directory.
 */
public class VoiceAssistant {
    // === Configure this ===
    private static final String MODEL_PATH = "C:\\models\\vosk-model-small-en-us-0.15"; // <-- update
    private static final float SAMPLE_RATE = 16000.0f;

    // FreeTTS voice name
    private static final String TTS_VOICE = "kevin16";

    // HTTP client and JSON mapper
    private final HttpClient httpClient = HttpClient.newHttpClient();
    private final ObjectMapper mapper = new ObjectMapper();

    private Voice ttsVoice;

    public static void main(String[] args) throws Exception {
        // Initialize bindings for Vosk native lib
        LibVosk.setLogLevel(LogLevel.INFO);
        VoiceAssistant assistant = new VoiceAssistant();
        assistant.initTTS();
        assistant.speak("Hello Sam. Voice assistant started and ready.");
        assistant.runRecognizerLoop();
    }

    private void initTTS() {
        System.setProperty("freetts.voices",
            "com.sun.speech.freetts.en.us.cmu_us_kal.KevinVoiceDirectory");
        VoiceManager vm = VoiceManager.getInstance();
        ttsVoice = vm.getVoice(TTS_VOICE);
        if (ttsVoice == null) {
            System.err.println("[TTS] Voice not found: " + TTS_VOICE);
            return;
        }
        ttsVoice.allocate();
    }

    private void speak(String text) {
        System.out.println("[Assistant] " + text);
        if (ttsVoice != null) {
            ttsVoice.speak(text);
        }
    }

    private void runRecognizerLoop() {
        // Open microphone line
        AudioFormat format = new AudioFormat(SAMPLE_RATE, 16, 1, true, false);
        DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);

        if (!AudioSystem.isLineSupported(info)) {
            System.err.println("[ERROR] Microphone not supported with format " + format);
            return;
        }

        try (Model model = new Model(MODEL_PATH)) {
            try (TargetDataLine line = (TargetDataLine) AudioSystem.getLine(info)) {
                line.open(format);
                line.start();

                try (InputStream ais = new AudioInputStream(line)) {
                    Recognizer recognizer = new Recognizer(model, SAMPLE_RATE);
                    byte[] buffer = new byte[4096];

                    speak("Listening now. Say a command like, 'what time is it', 'search', 'play', or 'open notepad'.");

                    while (true) {
                        int n = ais.read(buffer);
                        if (n < 0) break;

                        if (recognizer.acceptWaveForm(buffer, n)) {
                            String resultJson = recognizer.getResult();
                            String text = extractTextFromResult(resultJson);
                            if (text != null && !text.isBlank()) {
                                System.out.println("[Heard] " + text);
                                handleCommand(text);
                            }
                        } else {
                            // partial = recognizer.getPartialResult(); // optional
                        }
                    }
                    recognizer.close();
                }
            }
        } catch (LineUnavailableException e) {
            System.err.println("[ERROR] Microphone line unavailable: " + e.getMessage());
        } catch (IOException e) {
            System.err.println("[ERROR] Model load failed: " + e.getMessage());
        }
    }

    private String extractTextFromResult(String json) {
        // Vosk result JSON shape: {"text":"..."}
        try {
            Map<?,?> m = mapper.readValue(json, Map.class);
            Object t = m.get("text");
            return t == null ? "" : t.toString().trim().toLowerCase(Locale.ROOT);
        } catch (Exception e) {
            return "";
        }
    }

    private void handleCommand(String text) {
        try {
            if (text.contains("time") || text.matches(".*what.*time.*")) {
                String time = LocalTime.now().format(DateTimeFormatter.ofPattern("hh:mm a"));
                speak("The time is " + time);
            }
            else if (text.startsWith("search") || text.startsWith("search for") || text.startsWith("wiki")) {
                // e.g., "search tata motors" or "wiki albert einstein"
                String query = text.replaceFirst("search( for)?", "")
                                   .replaceFirst("wiki", "")
                                   .trim();
                if (query.isBlank()) {
                    speak("What would you like me to search?");
                    return;
                }
                speak("Searching Wikipedia for " + query);
                String summary = wikiSummary(query);
                if (summary != null && !summary.isBlank()) {
                    speak(summary);
                } else {
                    speak("I couldn't find a short summary for " + query);
                }
            }
            else if (text.startsWith("play")) {
                String song = text.replaceFirst("play", "").trim();
                if (song.isBlank()) {
                    speak("Which song should I play?");
                    return;
                }
                speak("Playing " + song + " on YouTube.");
                // Open default browser with YouTube search
                String url = "https://www.youtube.com/results?search_query=" + urlEncode(song);
                openBrowser(url);
            }
            else if (text.contains("open notepad") || text.contains("open notepad.exe") || text.contains("open notepad app")) {
                speak("Opening Notepad.");
                Runtime.getRuntime().exec("notepad.exe");
            }
            else if (text.contains("exit") || text.contains("quit") || text.contains("goodbye")) {
                speak("Goodbye, Sam.");
                System.exit(0);
            }
            else {
                speak("I don't know that command yet. Try 'what time', 'search', 'play', or 'open notepad'.");
            }
        } catch (Exception e) {
            speak("An error occurred handling the command.");
            e.printStackTrace();
        }
    }

    private String wikiSummary(String query) {
        try {
            // Wikipedia REST API summary endpoint
            String safe = urlEncode(query.replace(" ", "_"));
            String endpoint = "https://en.wikipedia.org/api/rest_v1/page/summary/" + safe;
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(endpoint))
                    .header("User-Agent", "SamVoiceAssistant/1.0 (contact: none)")
                    .GET()
                    .build();

            HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() == 200) {
                Map<?,?> m = mapper.readValue(resp.body(), Map.class);
                Object extract = m.get("extract");
                if (extract != null) {
                    String text = extract.toString();
                    // trim to first 2 sentences
                    int periodIdx = nthIndexOf(text, '.', 2);
                    if (periodIdx > 0 && periodIdx < text.length()-1) {
                        return text.substring(0, periodIdx+1);
                    } else {
                        return text;
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("[WIKI] error: " + e.getMessage());
        }
        return null;
    }

    // helpers
    private static String urlEncode(String s) {
        try {
            return java.net.URLEncoder.encode(s, java.nio.charset.StandardCharsets.UTF_8);
        } catch (Exception e) { return s; }
    }

    private static void openBrowser(String url) {
        try {
            if (Desktop.isDesktopSupported()) {
                Desktop.getDesktop().browse(new URI(url));
            } else {
                Runtime.getRuntime().exec(new String[]{"cmd", "/c", "start", url});
            }
        } catch (Exception e) {
            System.err.println("[BROWSER] could not open: " + e.getMessage());
        }
    }

    private static int nthIndexOf(String s, char c, int n) {
        int pos = -1;
        for (int i = 0; i < n; i++) {
            pos = s.indexOf(c, pos+1);
            if (pos == -1) return -1;
        }
        return pos;
    }
}
