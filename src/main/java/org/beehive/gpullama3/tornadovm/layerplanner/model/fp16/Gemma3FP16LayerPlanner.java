package org.beehive.gpullama3.tornadovm.layerplanner.model.fp16;

import org.beehive.gpullama3.inference.state.Gemma3State;
import org.beehive.gpullama3.inference.weights.tornado.Gemma3TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.gemma3.Gemma3Configuration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.FP16LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LlamaFP16FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.fp16.LogitsFP16Layer;

public class Gemma3FP16LayerPlanner extends FP16LayerPlanner<Gemma3State, Gemma3Configuration, Gemma3TornadoWeights> {

    public Gemma3FP16LayerPlanner(Gemma3State state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new Activation("activationUpdate", this.state, this.weights, this.config);
        this.ffnLayers = new LlamaFP16FFNLayers("gemma3FFN", this.state, this.weights, this.config, this.schedulerType);
        this.logitsLayer = new LogitsFP16Layer("gemma3Logits", this.state, this.weights, this.config, ffnLayers.getLastTaskGraphID(), this.schedulerType);
    }

}
