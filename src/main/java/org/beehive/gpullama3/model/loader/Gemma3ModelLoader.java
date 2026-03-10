package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tensor.GGUF;
import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tensor.tornado.FP32TornadoTensor;
import org.beehive.gpullama3.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.auxiliary.Pair;
import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Gemma3StandardWeights;
import org.beehive.gpullama3.inference.weights.tornado.Gemma3TornadoWeights;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.gemma3.Gemma3;
import org.beehive.gpullama3.model.gemma3.Gemma3Configuration;
import org.beehive.gpullama3.tokenizer.Gemma3Tokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tokenizer.Vocabulary;

import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

import java.nio.channels.FileChannel;
import java.util.Map;

import static org.beehive.gpullama3.model.loader.ModelLoader.*;

public class Gemma3ModelLoader extends AbstractModelLoader<Gemma3, Gemma3Configuration> {

    public Gemma3ModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return Vocabulary.loadGemma3Vocabulary(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        return new Gemma3Tokenizer(metadata, vocabulary);
    }

    // @formatter:off
    @Override
    protected Gemma3Configuration createConfiguration(Map<String, Object> metadata) {
        // Try gemma-specific metadata keys first, with llama fallbacks for compatibility
        String prefix = metadata.containsKey("gemma.embedding_length") ? "gemma" : "llama";

        int vocabSize = metadata.containsKey(prefix + ".vocab_size")
                ? (int) metadata.get(prefix + ".vocab_size")
                : (int) metadata.get("tokenizer.ggml.tokens.length");

        return new Gemma3Configuration(
                getModelQuantization(metadata),
                (int) metadata.get(prefix + ".embedding_length"),
                (int) metadata.get(prefix + ".feed_forward_length"),
                (int) metadata.get(prefix + ".block_count"),
                (int) metadata.get(prefix + ".attention.head_count"),
                metadata.containsKey(prefix + ".attention.head_count_kv") ?
                        (int) metadata.get(prefix + ".attention.head_count_kv")
                        : (int) metadata.get(prefix + ".attention.head_count"),
                vocabSize,
                (int) metadata.get(prefix + ".context_length"),
                (float) metadata.getOrDefault(prefix + ".attention.layer_norm_rms_epsilon", 1e-6f),
                (float) metadata.getOrDefault(prefix + ".rope.freq_base", 10000f)).withContextLength(contextLength);
    }
    // @formatter:on

    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(Gemma3Configuration config) {
        return RoPE.precomputeFreqsCis(config.contextLength(), config.dim() / config.numberOfHeads(), config.ropeTheta(), false, 1.0f, 1.0f, 1.0f, config.contextLength());
    }

    @Override
    protected Gemma3 createModel(Gemma3Configuration config, Tokenizer tokenizer, Weights weights) {
        return new Gemma3(config, tokenizer, weights, ChatFormat.create(tokenizer, null));
    }

    // @formatter:off
    @Override
    protected Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries, Gemma3Configuration config, Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
                                            GGMLTensorEntry outputWeight) {

        final int nl = config.numberOfLayers();

        return new Gemma3StandardWeights(
                loadTensor(tokenEmbeddings),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadTensor(tensorEntries.get("output_norm.weight")),
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                loadTensor(outputWeight),
                outputWeight.ggmlType());
    }
    // @formatter:on

    // @formatter:off
    @Override
    protected Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries,
                                             Gemma3Configuration config,
                                             Pair<float[], float[]> ropeFreqs,
                                             GGMLTensorEntry tokenEmbeddings,
                                             GGMLTensorEntry outputWeight) {
        GGMLType ggmlType = outputWeight.ggmlType();

        if (TornadoVMMasterPlan.ENABLE_TORNADOVM_INIT_TIME) {
            System.out.println("Loading model weights in TornadoVM format (loading " + ggmlType + ")");
        }

        // Validate supported types
        if (ggmlType != GGMLType.F16 && ggmlType != GGMLType.Q8_0) {
            throw new UnsupportedOperationException("Type: " + ggmlType + " currently not supported for TornadoVM weights.");
        }

        final int nl = config.numberOfLayers();

        // Load all tensors uniformly as TornadoTensor hierarchy
        return new Gemma3TornadoWeights(
                loadTornadoTensor(tokenEmbeddings),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),    // fp32
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),     // fp32
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")),
                loadArrayOfTornadoTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")),
                loadTornadoTensor(tensorEntries.get("output_norm.weight")),                                     // fp32
                new FP32TornadoTensor(FloatArray.fromArray(ropeFreqs.first())),
                new FP32TornadoTensor(FloatArray.fromArray(ropeFreqs.second())),
                loadTornadoTensor(outputWeight),
                ggmlType
        );
    }
    // @formatter:on
}
