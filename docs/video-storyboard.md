# Video Storyboard

## Creative direction

Start with a map: **attention and an MLP, repeated**. Then immediately locate those two lines in the code. This gives viewers a place to anchor each explanation that follows.

The strength of a video built around this repository is that **the structure just explained is right there in code that actually trains**. The goal is to start with an accessible explanation and end with viewers being able to read the real code.

## 1. Explain the two components through their contrasting roles

Pair the introductory explanations like this:

> Attention decides which information to bring in from earlier positions, and how much.
> The MLP processes that gathered information at each position.

Use shorter labels on screen:

- **Attention: bring in information from other positions**
- **MLP: process information at this position**

“Looking around the sentence” leaves it unclear what information is being retrieved. This is also a causal next-character model, so it cannot look at later characters.

The distinction matches the implementation: attention mixes information across positions, while the MLP applies the same neural network to the vector at each position. The MLP does not determine the next character on its own; the final prediction comes from the output matrix after the blocks.

Show this visually: **arrows connect characters during attention; during the MLP, the contents of each character's box change in place**.

## 2. The opening promise

Keep the confidence of “this is easy,” but make the promise specific: show viewers where to start reading the code.

### Opening option A

> Want to understand transformer code? Finding where to start is simple. Look for two things: attention and an MLP.

### Opening option B

> Transformer code looks complicated, right? Start by finding just two things: attention and an MLP.

The first minute should **help viewers start reading the code**, rather than promise complete understanding of everything.

## 3. The first-minute storyboard

Adjust the timing after a read-through, allowing room for the visuals.

| Time | Narration | Visuals |
|---|---|---|
| 0–7 sec | “Want to understand transformer code? Start by finding just two things.” | Show the actual code, then fade out the surrounding details. |
| 7–12 sec | “Attention, and an MLP.” | Introduce the two components in distinct colors. |
| 12–23 sec | “Attention decides which information to bring in from earlier positions, and how much.” | Connect the current character to earlier characters, using different line widths for different weights. |
| 23–32 sec | “The MLP processes that gathered information at each position.” | Remove the connections between characters; transform each position's vector in place. |
| 32–39 sec | “Bring information in, then process it. Repeat that pair several times.” | Stack four `Attention → MLP` blocks. |
| 39–48 sec | “Add the input, prediction, and learning from the correct answers, and one training step fits into ten lines.” | Show the README's ten-line overview on one screen. |
| 48–55 sec | “And the two components we just discussed are right here, in these two lines.” | Highlight `blk.attn(...)` and `blk.mlp(...)`. |
| 55–60 sec | “So what's inside `x`, and what changes as it passes through these lines?” | Zoom in on `x` and transition into the main explanation. |

End with a question. Rather than closing with “that's all there is to it,” leave the contents of `x` as the opening into the next section.

### What “ten lines” means

The ten lines are **an expanded overview of one training step**. The full implementation is 123 lines, with separate implementations of attention and the MLP. Use **“One training step, ten lines”** as the on-screen heading.

```python
tokens, answers = get_batch()
optimizer.zero_grad(set_to_none=True)
x = wte[tokens] + wpe[:T]
for blk in blocks:
    x = x + blk.attn(layer_norm(x))
    x = x + blk.mlp(layer_norm(x))
logits = layer_norm(x) @ wue
loss = cross_entropy(logits, answers)
loss.backward()
optimizer.step()
```

This is the explanatory overview from the README. In the actual file, the middle six lines are inside `Transformer.forward()`, and the remaining four are in `train()`. See [transformer.py](../transformer.py) for the executable version's function arguments and object references.

## 4. Keep returning to the code as a map

Rather than reading all ten lines aloud on their first appearance, keep these two lines in focus:

```python
for blk in blocks:
    x = x + blk.attn(layer_norm(x))
    x = x + blk.mlp(layer_norm(x))
```

Return to the same screen after each explanation in the main section:

- After explaining the input: `wte[tokens] + wpe[...]` makes sense.
- After explaining attention: `blk.attn(...)` makes sense.
- After explaining the MLP: `blk.mlp(...)` makes sense.
- After explaining residual connections: `x +` makes sense.
- After explaining training: `loss.backward()` and `optimizer.step()` make sense.

The payoff is **being able to read code that looked unfamiliar at the start**. Keep the colors for attention and the MLP consistent between the animation and the code.

## 5. Begin the main section with the goal: predict the next character

After the first minute, establish what the model is trying to do before diving into Q, K, and V.

> This code has one goal: look at the characters so far and predict the next one.

This repository uses a character-level model, so demonstrate it with individual characters to match the implementation.

```text
Input:   h e l l
Target:  e l l o
```

Use this scene to motivate the explanations that follow:

- Why attention? To use information from earlier positions.
- Why hide later positions? Because the model must not see the answer in advance.
- Why an output layer? To assign scores to possible next characters.
- Why training? To adjust the model so the correct character receives a higher score.

### Main-section sequence

**Next-character prediction → turn characters into vectors → attention → MLP → repetition and `x +` → prediction and training → actual generated text**

Introduce Q, K, and V as **“how we calculate which information to bring in from earlier characters, and how much”**, connecting back to the opening explanation.

## 6. A later twist: delete one line

Use the mask-removal experiment in [try-this.md](try-this.md). Deleting the line that hides future positions makes the loss drop dramatically.

In the documented 200-step experiment, training loss drops from roughly `2.54 → 0.74`. That happens because the model can now see answers already present in its input, rather than because it has become better at prediction. These numbers are quoted from the experiment recorded in the repository.

### Scene script

> Delete this one line, and the score gets much better.
>
> But did the model get smarter?
>
> No. It can now see the answer in advance.

This scene pays off the opening explanation that attention **only brings in information from earlier positions**, and makes the purpose of a single line of code tangible. Use it alongside actual generated text as a payoff in the later part of the video.

## Central line

> Attention brings in information from earlier positions. The MLP processes it at each position. At the heart of a transformer, these two operations repeat. Let's find them in the code.

Preserve the progression: **two components → two roles → repetition → two lines in real code**. The central aim is to make the structure promised in the opening readable in the actual implementation.

## Repository references

- [transformer.py](../transformer.py): the model, training, and generation implementation
- [README.md](../README.md): the ten-line training-step overview, how to run it, and generated samples
- [try-this.md](try-this.md): experiments removing the mask, attention, residual connections, and other components
- [line-mapping.md](line-mapping.md): correspondence with production model code
