package org.beehive.gpullama3.model.format;

import org.beehive.gpullama3.tokenizer.Gemma3Tokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Chat format implementation for Gemma 3 models.
 * <p>
 * Gemma 3 uses a turn-based chat format with the following structure:
 * <pre>
 * &lt;bos&gt;&lt;start_of_turn&gt;user
 * {user message}&lt;end_of_turn&gt;
 * &lt;start_of_turn&gt;model
 * {model response}&lt;end_of_turn&gt;
 * </pre>
 */
public class Gemma3ChatFormat implements ChatFormat {

    protected final Tokenizer tokenizer;
    protected final int beginOfText;
    protected final int startOfTurn;
    protected final int endOfTurn;
    protected final int endOfText;
    protected final Set<Integer> stopTokens;

    public Gemma3ChatFormat(Tokenizer tokenizer) {
        this.tokenizer = tokenizer;
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        this.beginOfText = specialTokens.getOrDefault("<bos>", 2);
        this.startOfTurn = specialTokens.getOrDefault("<start_of_turn>", 106);
        this.endOfTurn = specialTokens.getOrDefault("<end_of_turn>", 107);
        this.endOfText = specialTokens.getOrDefault("<eos>", 1);
        this.stopTokens = Set.of(endOfTurn, endOfText);
    }

    @Override
    public int getBeginOfText() {
        return beginOfText;
    }

    @Override
    public Set<Integer> getStopTokens() {
        return stopTokens;
    }

    @Override
    public List<Integer> encodeHeader(Message message) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startOfTurn);
        // Gemma uses "model" instead of "assistant" for the model role
        String roleName = message.role().equals(Role.ASSISTANT) ? "model" : message.role().name();
        tokens.addAll(tokenizer.encodeAsList(roleName));
        tokens.addAll(tokenizer.encodeAsList("\n"));
        return tokens;
    }

    @Override
    public List<Integer> encodeMessage(Message message) {
        List<Integer> tokens = encodeHeader(message);
        tokens.addAll(tokenizer.encodeAsList(message.content().strip()));
        tokens.add(endOfTurn);
        tokens.addAll(tokenizer.encodeAsList("\n"));
        return tokens;
    }

    public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<Message> dialog) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(beginOfText);
        for (Message message : dialog) {
            tokens.addAll(encodeMessage(message));
        }
        if (appendAssistantTurn) {
            tokens.addAll(encodeHeader(new Message(ChatFormat.Role.ASSISTANT, "")));
        }
        return tokens;
    }
}
