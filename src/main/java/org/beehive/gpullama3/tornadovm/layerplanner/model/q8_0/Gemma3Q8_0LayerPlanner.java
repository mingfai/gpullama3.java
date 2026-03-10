package org.beehive.gpullama3.tornadovm.layerplanner.model.q8_0;

import org.beehive.gpullama3.inference.state.Gemma3State;
import org.beehive.gpullama3.inference.weights.tornado.Gemma3TornadoWeights;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.gemma3.Gemma3Configuration;
import org.beehive.gpullama3.tornadovm.layerplanner.quantization.Q8_0LayerPlanner;
import org.beehive.gpullama3.tornadovm.layers.Activation;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.Gemma3Q8_0FFNLayers;
import org.beehive.gpullama3.tornadovm.layers.type.q8_0.LogitsQ8_0Layer;

public class Gemma3Q8_0LayerPlanner extends Q8_0LayerPlanner<Gemma3State, Gemma3Configuration, Gemma3TornadoWeights> {

    public Gemma3Q8_0LayerPlanner(Gemma3State state, Model model) {
        super(state, model);
        validateQuantizationType();
        setupTornadoForwardPlan();
    }

    @Override
    protected void initializeLayerComponents() {
        this.activationLayer = new Activation("activationUpdate", this.state, this.weights, this.config);
        this.ffnLayers = new Gemma3Q8_0FFNLayers("gemma3FFN", this.state, this.weights, this.config, this.schedulerType);
        this.logitsLayer = new LogitsQ8_0Layer("gemma3Logits", this.state, this.weights, this.config, ffnLayers.getLastTaskGraphID(), this.schedulerType);
    }

}
