# ten-line-transformer

[한국어](README.md)

A transformer in one Python file. **123 lines**, with training, validation and
generation all in it. No module-level variables and no comments — the function
names stand in for the comments.

A book in progress, *Reading a Transformer in Ten Lines* (working title, Korean),
reads this file from the first line to the last. The code stands on its own
without the book.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YoungjuneKwon/ten-line-transformer/blob/main/notebook.ipynb)

## One training step is ten lines

```python
tokens, answers = get_batch()               # (B, T), (B, T)
optimizer.zero_grad(set_to_none=True)
x = wte[tokens] + wpe[:T]                   # (B, T, C)
for blk in blocks:                          # n_block times
    x = x + blk.attn(layer_norm(x))         # (B, T, C)
    x = x + blk.mlp(layer_norm(x))          # (B, T, C)
logits = layer_norm(x) @ wue                # (B, T, V)
loss = cross_entropy(logits, answers)       # one scalar
loss.backward()
optimizer.step()
```

In the actual file the middle six lines sit in `Transformer.forward()` and the
other four in `train()`.

## Running it

The Colab badge above needs nothing but a browser. On your own machine you need
PyTorch and one text file.

```bash
pip install torch
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python transformer.py 200 input.txt
```

It runs on CPU. No graphics card required.

## You get the same numbers

The seed is fixed at 1337. Run it to 200 steps and the loss comes down like this.

```
  step     train       val
     0    4.2006    3.9836
   100    2.7746    2.7454
   200    2.5400    2.4785
```

How long it takes differs by machine. The loss values do not.

**If your numbers differ, please open an issue.** That is the one thing this
repository most wants to hear about — see [CONTRIBUTING.md](CONTRIBUTING.md).

## The model

```
B=32  T=64  C=128  H=4  n_block=4      19 learned tensors, 811,264 numbers
wte (65,128)  wpe (64,128)  wue (128,65)
blocks.N.attn.qkv (384,128)  attn.proj (128,128)  mlp.fc (512,128)  mlp.out (128,512)
```

Biases, dropout, learned normalization coefficients and weight tying are
**left out on purpose.** They are not needed to show the structure, and leaving
them out leaves less to read.

## Where it came from

Rewritten substantially in PyTorch on top of
[nanoGPT](https://github.com/karpathy/nanoGPT), which is public online.
Compared character by character, 1.3% of the whole file and 22.6% of the
attention body overlap. But **the order of the computation is the same ten
steps.**

```
qkv projection -> split in 3 -> separate heads -> q·kᵀ -> divide by √
        -> mask -> softmax -> @ v -> merge heads -> output projection
```

The result was checked against the original at every step of the rewrite. Run
200 steps with the same seed and the 811,264 learned numbers do not differ by
**a single bit** (with threads pinned to one. With several threads the addition
order changes and the last digit wobbles).

A line-by-line map of what came from where is in [NOTICE.md](NOTICE.md).

## License

MIT. Third-party attribution is in [NOTICE.md](NOTICE.md).

Experiment code and measurement logs will be published together with the book.
