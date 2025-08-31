import java.io.*;
import java.util.*;
import org.apache.commons.math3.linear.*;

public class Chatbot {
    private List<String> corpus = new ArrayList<>(); // corpus of responses
    private List<String> docs = new ArrayList<>(); // corpus text (questions/faq)
    private Map<String, Integer> vocab = new HashMap<>();
    private List<RealVector> tfidfVectors = new ArrayList<>();
    private int vocabSize = 0;

    public void loadCorpus(String path) throws Exception {
        // simple format: each line is "question \t answer"
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;
        while ((line = br.readLine()) != null) {
            String[] parts = line.split("\t");
            if (parts.length >= 2) {
                docs.add(parts[0].toLowerCase());
                corpus.add(parts[1]);
            }
        }
        br.close();
        buildVocab();
        computeTFIDF();
    }

    private List<String> tokenize(String s) {
        return Arrays.asList(s.replaceAll("[^a-z0-9 ]", " ").split("\\s+"));
    }

    private void buildVocab() {
        Set<String> words = new LinkedHashSet<>();
        for (String d : docs) {
            words.addAll(tokenize(d));
        }
        int idx = 0;
        for (String w : words) {
            if (!w.isBlank()) {
                vocab.put(w, idx++);
            }
        }
        vocabSize = vocab.size();
    }

    private RealVector toTfVector(String doc) {
        double[] vec = new double[vocabSize];
        List<String> toks = tokenize(doc);
        for (String t : toks) {
            if (vocab.containsKey(t)) vec[vocab.get(t)] += 1.0;
        }
        return new ArrayRealVector(vec);
    }

    private void computeTFIDF() {
        int n = docs.size();
        double[] df = new double[vocabSize];
        List<RealVector> tfs = new ArrayList<>();
        for (String d : docs) {
            RealVector v = toTfVector(d);
            tfs.add(v);
            for (int i = 0; i < vocabSize; i++) if (v.getEntry(i) > 0) df[i] += 1.0;
        }
        for (RealVector tf : tfs) {
            double[] arr = new double[vocabSize];
            for (int i = 0; i < vocabSize; i++) {
                double tfval = tf.getEntry(i);
                double idf = Math.log((n + 1.0) / (df[i] + 1.0)) + 1.0;
                arr[i] = tfval * idf;
            }
            tfidfVectors.add(new ArrayRealVector(arr));
        }
    }

    private double cosine(RealVector a, RealVector b) {
        double denom = a.getNorm() * b.getNorm();
        if (denom == 0) return 0.0;
        return a.dotProduct(b) / denom;
    }

    public String respond(String input) {
        RealVector qv = toTfVector(input.toLowerCase());
        // convert qv to tf-idf using same idf as training docs
        double[] arr = new double[vocabSize];
        int n = docs.size();
        double[] df = new double[vocabSize];
        for (String d : docs) {
            RealVector v = toTfVector(d);
            for (int i = 0; i < vocabSize; i++) if (v.getEntry(i) > 0) df[i] += 1.0;
        }
        for (int i = 0; i < vocabSize; i++) {
            double idf = Math.log((n + 1.0) / (df[i] + 1.0)) + 1.0;
            arr[i] = qv.getEntry(i) * idf;
        }
        RealVector qtfidf = new ArrayRealVector(arr);
        double bestScore = -1; int bestIdx = -1;
        for (int i = 0; i < tfidfVectors.size(); i++) {
            double s = cosine(qtfidf, tfidfVectors.get(i));
            if (s > bestScore) { bestScore = s; bestIdx = i; }
        }
        if (bestIdx == -1 || bestScore < 0.05) return "Sorry, I don't know the answer to that yet.";
        return corpus.get(bestIdx);
    }

    public static void main(String[] args) throws Exception {
        Chatbot bot = new Chatbot();
        bot.loadCorpus("qa_corpus.txt"); // each line: "question\tanswer"
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        System.out.println("Chatbot ready. Type 'exit' to quit.");
        while (true) {
            System.out.print("You: ");
            String line = br.readLine();
            if (line == null || line.equalsIgnoreCase("exit")) break;
            String ans = bot.respond(line);
            System.out.println("Bot: " + ans);
        }
    }
}
