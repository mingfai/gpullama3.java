package org.beehive.gpullama3.tokenizer;

import org.beehive.gpullama3.auxiliary.Pair;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * GPT-2-style BPE tokenizer for Gemma 3 models.
 * <p>
 * Gemma 3 uses a SentencePiece-based BPE tokenizer with byte-level encoding,
 * similar to the Llama tokenizer but with Gemma-specific special tokens
 * and pretokenization patterns.
 */
public class Gemma3Tokenizer implements Tokenizer {
    static final Map<Integer, Integer> BYTE_ENCODER = bytesToUnicode();
    static final Map<Integer, Integer> BYTE_DECODER = BYTE_ENCODER.entrySet().stream().collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));
    private static final String GEMMA_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    // general fields
    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    // model-specific fields
    private final Map<Pair<Integer, Integer>, Integer> merges;
    private final Map<String, Integer> specialTokens;

    public Gemma3Tokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        // load from metadata
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges = Arrays.stream(mergeLines).map(line -> line.split(" "))
                .map(parts -> new Pair<>(vocabulary.getIndex(parts[0]).orElseThrow(), vocabulary.getIndex(parts[1]).orElseThrow())).toList();
        int allTokens = vocabulary.size();

        // Gemma models typically have special tokens at the beginning and end of vocabulary
        // Detect base token count from metadata or use vocabulary size
        int baseTokens = allTokens;
        Object preTokenType = metadata.get("tokenizer.ggml.pre");
        if (preTokenType != null) {
            // If pre-tokenizer info is available, use it to determine base tokens
            baseTokens = allTokens;
        }

        // Build special tokens map from GGUF metadata token types
        Map<String, Integer> specialTokens = new HashMap<>();
        int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
        if (tokenTypes != null) {
            for (int i = 0; i < tokenTypes.length && i < allTokens; i++) {
                // Token type 3 = control/special token in GGUF format
                if (tokenTypes[i] == 3) {
                    specialTokens.put(vocabulary.get(i), i);
                }
            }
        }

        // Ensure essential Gemma tokens are present
        if (!specialTokens.containsKey("<bos>")) {
            vocabulary.getIndex("<bos>").ifPresent(idx -> specialTokens.put("<bos>", idx));
        }
        if (!specialTokens.containsKey("<eos>")) {
            vocabulary.getIndex("<eos>").ifPresent(idx -> specialTokens.put("<eos>", idx));
        }
        if (!specialTokens.containsKey("<start_of_turn>")) {
            vocabulary.getIndex("<start_of_turn>").ifPresent(idx -> specialTokens.put("<start_of_turn>", idx));
        }
        if (!specialTokens.containsKey("<end_of_turn>")) {
            vocabulary.getIndex("<end_of_turn>").ifPresent(idx -> specialTokens.put("<end_of_turn>", idx));
        }

        // init tokenizer object fields
        this.vocabulary = vocabulary;
        this.compiledPattern = Pattern.compile(GEMMA_PATTERN);
        this.specialTokens = new HashMap<>(specialTokens);
        this.merges = new HashMap<>();
        for (Pair<Integer, Integer> pair : merges) {
            int firstIndex = pair.first();
            int secondIndex = pair.second();
            int mergeIndex = vocabulary.getIndex(vocabulary.get(firstIndex) + vocabulary.get(secondIndex)).orElseThrow();
            this.merges.put(pair, mergeIndex);
        }
    }

    private static List<String> findAll(Pattern pattern, String text) {
        List<String> allMatches = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            allMatches.add(matcher.group());
        }
        return allMatches;
    }

    private static List<Integer> merge(List<Integer> ids, Pair<Integer, Integer> pair, int idx) {
        List<Integer> newids = new ArrayList<>();
        int i = 0;
        while (i < ids.size()) {
            if (ids.get(i).equals(pair.first()) && i < ids.size() - 1 && ids.get(i + 1).equals(pair.second())) {
                newids.add(idx);
                i += 2;
            } else {
                newids.add(ids.get(i));
                i += 1;
            }
        }
        return newids;
    }

    private static Map<Integer, Integer> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>();
        IntStream.rangeClosed('!', '~').forEach(bs::add);
        IntStream.rangeClosed('¡', '¬').forEach(bs::add);
        IntStream.rangeClosed('®', 'ÿ').forEach(bs::add);

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n += 1;
            }
        }

        return IntStream.range(0, bs.size()).boxed().collect(Collectors.toMap(bs::get, cs::get));
    }

    public String regexPattern() {
        if (compiledPattern == null) {
            return null;
        }
        return compiledPattern.pattern();
    }

    @Override
    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }

    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return specialTokens.containsValue(tokenIndex);
    }

    @Override
    public boolean shouldDisplayToken(int token) {
        return !isSpecialToken(token);
    }

    private int[] encodeImpl(String text) {
        return encode(text, Set.of()).stream().mapToInt(i -> i).toArray();
    }

    public List<Integer> encode(String text, Set<String> allowedSpecial) {
        Set<String> special = allowedSpecial;
        assert getSpecialTokens().keySet().containsAll(special);
        if (special.isEmpty()) {
            return encodeOrdinary(text);
        }

        String specialPattern = special.stream().map(Pattern::quote).collect(Collectors.joining("|", "(", ")"));

        String[] specialChunks = text.split(specialPattern);
        List<Integer> ids = new ArrayList<>();
        for (String part : specialChunks) {
            if (special.contains(part)) {
                ids.add(getSpecialTokens().get(part));
            } else {
                ids.addAll(encodeOrdinary(part));
            }
        }
        return ids;
    }

    public List<Integer> encodeOrdinary(String text) {
        List<String> textChunks = findAll(compiledPattern, text);
        List<Integer> ids = new ArrayList<>();
        for (String chunk : textChunks) {
            List<Integer> chunkIds = encodeChunk(chunk);
            ids.addAll(chunkIds);
        }
        return ids;
    }

    private Map<Pair<Integer, Integer>, Integer> getStats(List<Integer> ids) {
        Map<Pair<Integer, Integer>, Integer> map = new HashMap<>();
        for (int i = 0; i + 1 < ids.size(); i++) {
            Pair<Integer, Integer> key = new Pair<>(ids.get(i), ids.get(i + 1));
            map.put(key, map.getOrDefault(key, 0) + 1);
        }
        return map;
    }

    private List<Integer> encodeChunk(String chunk) {
        List<Integer> ids = new ArrayList<>();
        for (int b : chunk.toCharArray()) {
            int tokenIndex = this.vocabulary.getIndex(String.valueOf((char) b)).orElseThrow();
            ids.add(tokenIndex);
        }

        while (ids.size() >= 2) {
            Map<Pair<Integer, Integer>, Integer> stats = getStats(ids);
            Pair<Integer, Integer> pair = stats.keySet().stream().min(Comparator.comparingInt(key -> this.merges.getOrDefault(key, Integer.MAX_VALUE))).orElseThrow();
            if (!this.merges.containsKey(pair)) {
                break;
            }
            int idx = this.merges.get(pair);
            ids = merge(ids, pair, idx);
        }
        return ids;
    }

    public String decodeImpl(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.get(token);
            sb.append(tokenString);
        }
        return sb.toString();
    }

    public int[] encode(String text) {
        StringBuilder sb = new StringBuilder();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
            sb.appendCodePoint(BYTE_ENCODER.get(Byte.toUnsignedInt(b)));
        }
        return encodeImpl(sb.toString());
    }

    @Override
    public List<Integer> encodeAsList(String text) {
        StringBuilder sb = new StringBuilder();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
            sb.appendCodePoint(BYTE_ENCODER.get(Byte.toUnsignedInt(b)));
        }
        return Arrays.stream(encodeImpl(sb.toString())).boxed().toList();
    }

    @Override
    public String decode(List<Integer> tokens) {
        String decoded = decodeImpl(tokens);
        int[] decodedBytesAsInts = decoded.codePoints().map(BYTE_DECODER::get).toArray();
        byte[] rawBytes = new byte[decodedBytesAsInts.length];
        for (int i = 0; i < decodedBytesAsInts.length; i++) {
            rawBytes[i] = (byte) decodedBytesAsInts[i];
        }
        return new String(rawBytes, StandardCharsets.UTF_8);
    }
}
