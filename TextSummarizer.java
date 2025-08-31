import java.util.*;
import java.util.stream.Collectors;

public class TextSummarizer {

    private static List<String> splitSentences(String text) {
        // naive split (replace with OpenNLP for production)
        return Arrays.stream(text.split("(?<=[.!?])\\s+"))
                .map(String::trim).filter(s -> !s.isEmpty()).collect(Collectors.toList());
    }

    private static List<String> tokenize(String s) {
        return Arrays.stream(s.toLowerCase().replaceAll("[^a-z0-9 ]", " ").split("\\s+"))
                .filter(t -> t.length() > 0).collect(Collectors.toList());
    }

    private static double sentenceSimilarity(String s1, String s2) {
        List<String> t1 = tokenize(s1), t2 = tokenize(s2);
        if (t1.isEmpty() || t2.isEmpty()) return 0.0;
        Set<String> words = new HashSet<>();
        words.addAll(t1); words.addAll(t2);
        double[] v1 = new double[words.size()], v2 = new double[words.size()];
        int idx = 0;
        Map<String, Integer> idxMap = new HashMap<>();
        for (String w : words) idxMap.put(w, idx++);
        for (String w : t1) v1[idxMap.get(w)]++;
        for (String w : t2) v2[idxMap.get(w)]++;
        // cosine
        double dot=0, n1=0, n2=0;
        for (int i=0;i<v1.length;i++) { dot += v1[i]*v2[i]; n1 += v1[i]*v1[i]; n2 += v2[i]*v2[i]; }
        if (n1==0 || n2==0) return 0.0;
        return dot / (Math.sqrt(n1)*Math.sqrt(n2));
    }

    private static double[] pagerank(double[][] M, double d, int iter) {
        int N = M.length;
        double[] pr = new double[N];
        Arrays.fill(pr, 1.0 / N);
        for (int it = 0; it < iter; it++) {
            double[] newpr = new double[N];
            for (int i = 0; i < N; i++) {
                double sum = 0;
                for (int j = 0; j < N; j++) {
                    double out = 0;
                    double rowSum = 0;
                    for (int k = 0; k < N; k++) rowSum += M[j][k];
                    if (rowSum != 0) out = M[j][i] / rowSum;
                    sum += pr[j] * out;
                }
                newpr[i] = (1 - d) / N + d * sum;
            }
            pr = newpr;
        }
        return pr;
    }

    public static String summarize(String text, int numSentences) {
        List<String> sentences = splitSentences(text);
        int N = sentences.size();
        if (N <= numSentences) return text;
        double[][] sim = new double[N][N];
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++)
            sim[i][j] = sentenceSimilarity(sentences.get(i), sentences.get(j));
        double[] scores = pagerank(sim, 0.85, 50);
        // pick top indices
        Integer[] idx = new Integer[N];
        for (int i = 0;i<N;i++) idx[i]=i;
        Arrays.sort(idx, (a,b) -> Double.compare(scores[b], scores[a]));
        Arrays.sort(Arrays.copyOf(idx, numSentences)); // to preserve order - we'll sort selected later
        Set<Integer> selected = new TreeSet<>();
        for (int i=0;i<numSentences;i++) selected.add(idx[i]);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < N; i++) {
            if (selected.contains(i)) {
                sb.append(sentences.get(i)).append(" ");
            }
        }
        return sb.toString().trim();
    }

    // quick demo
    public static void main(String[] args) {
        String text = "Artificial Intelligence is one of the most transformative technologies of the 21st century. "
            + "It is impacting industries from healthcare to finance, changing how we live and work. "
            + "However, AI also brings challenges such as ethical concerns, job displacement, and bias in algorithms. "
            + "Balancing innovation with responsibility is key for a sustainable AI future.";
        String summary = summarize(text, 2);
        System.out.println("Summary:\n" + summary);
    }
}
