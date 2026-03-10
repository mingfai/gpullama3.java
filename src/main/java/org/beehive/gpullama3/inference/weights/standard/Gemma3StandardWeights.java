package org.beehive.gpullama3.inference.weights.standard;

import org.beehive.gpullama3.tensor.GGMLType;
import org.beehive.gpullama3.tensor.standard.FloatTensor;

/**
 * A model-specific implementation of {@link StandardWeights} for the Gemma 3 model.
 * This class encapsulates the weights required for performing inference
 * using the Gemma 3 model in the standard CPU-based format.
 */
public class Gemma3StandardWeights extends StandardWeights {

    // @formatter:off
    public Gemma3StandardWeights(
            FloatTensor token_embedding_table,
            FloatTensor[] rms_att_weight,
            FloatTensor[] wq,
            FloatTensor[] wk,
            FloatTensor[] wv,
            FloatTensor[] wo,
            FloatTensor[] rms_ffn_weight,
            FloatTensor[] w1,
            FloatTensor[] w2,
            FloatTensor[] w3,
            FloatTensor rms_final_weight,
            FloatTensor freq_cis_real,
            FloatTensor freq_cis_imag,
            FloatTensor wcls,
            GGMLType weightType) {
        super(token_embedding_table,
                rms_att_weight,
                wq,
                wk,
                wv,
                wo,
                rms_ffn_weight,
                w1,
                w2,
                w3,
                rms_final_weight,
                freq_cis_real,
                freq_cis_imag,
                wcls,
                weightType);
    }
    // @formatter:on

    @Override
    public GGMLType getWeightType() {
        return weightType;
    }
}
