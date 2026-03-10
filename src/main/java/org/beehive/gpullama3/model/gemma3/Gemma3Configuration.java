package org.beehive.gpullama3.model.gemma3;

import org.beehive.gpullama3.model.Configuration;

// @formatter:off
public record Gemma3Configuration(String quantization,
                                  int dim,
                                  int hiddenDim,
                                  int numberOfLayers,
                                  int numberOfHeads,
                                  int numberOfKeyValueHeads,
                                  int vocabularySize,
                                  int contextLength,
                                  float rmsNormEps,
                                  float ropeTheta) implements Configuration {

    @Override
    public String quantization() {
        return quantization;
    }

    @Override
    public int numberOfHeadsKey() {
        return numberOfKeyValueHeads;
    }

    @Override
    public int contextLengthModel() {
        return contextLength;
    }

    /** Size of each attention head (derived from dim / numberOfHeads) */
    public int headSize() {
        return dim / numberOfHeads;
    }

    /** Key/value dimension (derived from dim * numberOfKeyValueHeads / numberOfHeads) */
    public int kvDim() {
        return dim * numberOfKeyValueHeads / numberOfHeads;
    }

    /** Multiplier for key/value sharing in grouped-query attention */
    public int kvMul() {
        return numberOfHeads / numberOfKeyValueHeads;
    }

    /**
     * Creates a new Configuration with a different context length.
     *
     * @param newContextLength The new context length to use
     * @return A new Configuration instance with updated context length,
     *         or the current instance if newContextLength is negative
     */
    // @formatter:off
    public Gemma3Configuration withContextLength(int newContextLength) {
        if (newContextLength < 0) {
            return this; // no change
        }
        return new Gemma3Configuration(
                this.quantization,
                this.dim,
                this.hiddenDim,
                this.numberOfLayers,
                this.numberOfHeads,
                this.numberOfKeyValueHeads,
                this.vocabularySize,
                newContextLength,
                this.rmsNormEps,
                this.ropeTheta
        );
    }
    // @formatter:on
}
