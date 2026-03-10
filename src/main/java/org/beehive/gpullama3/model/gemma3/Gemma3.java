package org.beehive.gpullama3.model.gemma3;

import org.beehive.gpullama3.inference.InferenceCore;
import org.beehive.gpullama3.inference.InferenceEngine;
import org.beehive.gpullama3.inference.sampler.Sampler;
import org.beehive.gpullama3.inference.state.Gemma3State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.model.AbstractModel;
import org.beehive.gpullama3.model.ModelType;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.tokenizer.Gemma3Tokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;

import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

public class Gemma3 extends AbstractModel {

    private final Gemma3Configuration configuration;

    public Gemma3(Gemma3Configuration configuration, Tokenizer tokenizer, Weights weights, ChatFormat chatFormat) {
        super(tokenizer, weights, chatFormat, null);
        this.configuration = configuration;
    }

    @Override
    public Gemma3Configuration configuration() {
        return configuration;
    }

    @Override
    public Gemma3Tokenizer tokenizer() {
        return (Gemma3Tokenizer) tokenizer;
    }

    @Override
    public ModelType getModelType() {
        return ModelType.GEMMA_3;
    }

    @Override
    public State createNewState() {
        State state = new Gemma3State(configuration(), -1);
        // Gemma uses <bos> token (typically token ID 2) as BOS
        state.latestToken = tokenizer.getSpecialTokens().getOrDefault("<bos>", 2);
        return state;
    }

    @Override
    public State createNewState(int batchsize) {
        State state = new Gemma3State(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().getOrDefault("<bos>", 2);
        return state;
    }

    @Override
    public void forward(State state, int token, int position) {
        InferenceCore.forwardJavaGemma3(this, state, token, position);
    }

    @Override
    public List<Integer> generateTokens(State state, int startPosition, List<Integer> promptTokens,
            Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        return InferenceEngine.generateTokensGemma3(this, state, startPosition, promptTokens,
                stopTokens, maxTokens, sampler, echo, onTokenGenerated);
    }

    @Override
    public List<Integer> generateTokensGPU(State state, int startPosition, List<Integer> promptTokens,
            Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated, TornadoVMMasterPlan tornadoVMPlan) {
        return InferenceEngine.generateTokensGPUGemma3(this, state, startPosition, promptTokens,
                stopTokens, maxTokens, sampler, echo, onTokenGenerated, tornadoVMPlan);
    }
}
