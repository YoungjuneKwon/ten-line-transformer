# Try this — six edits, five minutes each

Reading the file tells you what it does. Breaking it tells you what each part is
*for*. These are the six edits that taught me the most, in the order I would do
them.

Every number below was measured on one CPU-only machine with the command the
README uses, `OMP_NUM_THREADS=1 python transformer.py 200 input.txt`, except
where a 2000-step run is named. The timings differ by machine. The loss values
should not — if yours do,
[open an issue](https://github.com/YoungjuneKwon/ten-line-transformer/issues/new?template=1-numbers-differ.yml).

| edit | parameters | train @200 | val @200 |
|---|---:|---:|---:|
| **none — the file as it ships** | 811,264 | 2.5400 | 2.4785 |
| delete the mask line (L38) | 811,264 | **0.7399** | **0.6884** |
| delete the attention line (L69) | 811,264 | 2.5807 | 2.5387 |
| drop the residual (`x = x +` → `x =`) | 811,264 | **3.3648** | **3.3450** |
| `C` 128 → 64 | 209,024 | 2.7493 | 2.7361 |
| `n_block` 4 → 1 | 221,440 | 2.5539 | 2.5480 |
| seed 1337 → 1338 | 811,264 | 2.4934 | 2.5139 |

---

## 1. Delete the mask and watch the loss get *better*

[Line 38](../transformer.py#L38). Delete it.

```python
att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
```

```
            train     val
before     2.5400   2.4785
after      0.7399   0.6884      <- three times better
```

A loss that good would be a result worth publishing, and it is worth nothing.
Without that line every position can see every other position, including the one
it is being asked to predict. The model is not predicting anything; it is
copying the answer out of its own input.

**This is the most useful five minutes in the repository.** It is the clearest
demonstration I know that a loss number means nothing without knowing what the
model was allowed to see — and it is one line.

## 2. Delete attention, and learn to distrust short runs

[Line 69](../transformer.py#L69). Delete it, so each block is MLP only.

```python
x = x + blk.attn(layer_norm(x))
```

At 200 steps, the result looks like attention does not matter.

```
            train     val
before     2.5400   2.4785
after      2.5167   2.5717      <- a tie, near enough
```

Now run both for 2000 steps.

```
step        with attention        without
   0      4.2006   3.9836      4.2149   4.0473
 200      2.4812   2.5206      2.5167   2.5717
1000      1.8387   1.9600      2.4606   2.4980
2000      1.6408   1.8684      2.4754   2.5350      <- stuck since step 600
```

The model without attention **stops improving at about 2.45** and stays there.
The full model keeps going. Attention's contribution is invisible at 200 steps
and decisive at 2000.

Keep that result. It is the reason the 200-step run in the README is presented
as a reproducibility check and never as evidence that something works.

## 3. Remove the residual and watch it struggle

[Lines 69–70](../transformer.py#L69-L70). Change both `x = x +` into `x =`.

```python
x = blk.attn(layer_norm(x))
x = blk.mlp(layer_norm(x))
```

```
            train     val
before     2.5400   2.4785
after      3.3648   3.3450
```

Same parameters, same data, same steps — and it has barely left where it
started (4.20). Each block now **replaces** what came in instead of adding to
it, so nothing survives four blocks intact and the gradient has no short path
back. Two plus signs are the difference.

## 4. Halve the width

[Line 111](../transformer.py#L111): `C` from 128 to 64.

```python
B, T, C, H, n_block = 32, 64, 64, 4, 4
```

```
parameters   811,264  ->  209,024      a quarter
train @200    2.5400  ->   2.7493
```

A quarter of the parameters costs about 0.21 of loss here. Nobody derived 128;
it was chosen, and this is how you check what the choice bought. Try 256 too —
the loss goes lower and the wait gets much longer.

## 5. Take away three of the four blocks

[Line 111](../transformer.py#L111): `n_block` from 4 to 1.

```python
B, T, C, H, n_block = 32, 64, 128, 4, 1
```

```
parameters   811,264  ->  221,440
train @200    2.5400  ->   2.5539      almost identical
```

After experiment 2 you should refuse to accept that without a longer run. So
here is the longer run, and this time it does **not** overturn the short one.

```
step        4 blocks              1 block
   0      4.2006   3.9836      4.2032   4.0545
1000      1.8387   1.9600      1.8907   1.9930
2000      1.6408   1.8684      1.6922   1.9528
wall      265 s                 102 s
```

One block gets within 0.05 of four, on a quarter of the parameters, in 40% of
the time. At this size depth is nearly free to give up — which is not what I
expected to find, and is the honest answer.

Put experiments 2 and 5 side by side and that is the whole lesson: the short run
hid attention completely and reported depth correctly. **You cannot tell which
case you are in without running it.** That is the habit this file is trying to
build, and it is cheaper to build here than on a model that costs money.

## 6. Change the seed

[Line 113](../transformer.py#L113): `torch.manual_seed(1338)`.

```
            train     val
1337       2.5400   2.4785
1338       2.4934   2.5139
```

The numbers move by about 0.05 and the shape of the curve does not. Worth doing
once, so that when you compare two ideas you already know how big a difference
has to be before it means anything. A change smaller than this gap is not a
result.

---

## Putting one back

```bash
git checkout transformer.py
```

Everything above is a single-file edit, so this undoes any of it.

## If something surprised you

That is the repository working. Two places to put it:

- a number that does not match → [the reproduction form](https://github.com/YoungjuneKwon/ten-line-transformer/issues/new?template=1-numbers-differ.yml)
- a question about *why* → [Discussions](https://github.com/YoungjuneKwon/ten-line-transformer/discussions)
