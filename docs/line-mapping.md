# Line by line, against the official sources

`scripts/make_line_mapping.py` writes this file. Do not edit it by hand.

Every row links to a line in a repository that is not mine. The outer
columns are production model code in
[huggingface/transformers](https://github.com/huggingface/transformers),
pinned at tag `v5.18.0` so the line numbers keep pointing at the same code.
Click a link and you are looking at the real file.

**The rows that say *not present* are the point as much as the rest.**
`transformer.py` is 123 lines and a production file is around
500, and this is where the difference goes: building the attention mask,
sharing key and value across heads, rotary positions, caching, config.
None of it changes the order of the computation.

## Sources

- **ours** — [`transformer.py`](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py) · 123 lines ·
  sha256 `385bc1cfddec38c0`
- **transformers** — tag `v5.18.0`, https://github.com/huggingface/transformers

If a row is wrong, incomplete, or flatters this file, that is a bug —
please open an issue.

## Qwen2.5 — `src/transformers/models/qwen2/modeling_qwen2.py` (501 lines)

| place | ours (123 lines) | Qwen2.5 |
|---|---|---|
| normalization | [L18](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L18) | [L238-L255](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen2/modeling_qwen2.py#L238-L255) |
| positions | [L62](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L62) | [L113-L135](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen2/modeling_qwen2.py#L113-L135) |
| q, k and v in one matrix | [L29](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L29) | [L176-L234](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen2/modeling_qwen2.py#L176-L234) |
| score table `q @ kᵀ` | [L37](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L37) | [L163](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen2/modeling_qwen2.py#L163) |
| add the mask | [L38](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L38) | [L165](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen2/modeling_qwen2.py#L165) |
| build the mask | — (not present) | [L378](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen2/modeling_qwen2.py#L378) |
| softmax | [L39](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L39) | [L167](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen2/modeling_qwen2.py#L167) |
| weighted average of v | [L39](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L39) | [L169](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen2/modeling_qwen2.py#L169) |
| shared heads (GQA) | — (not present) | [L138-L147](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen2/modeling_qwen2.py#L138-L147) |
| MLP | [L50](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L50) | [L35-L48](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen2/modeling_qwen2.py#L35-L48) |
| the two residual lines | [L69-L70](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L69-L70) | [L291-L297](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen2/modeling_qwen2.py#L291-L297) |
| embedding | [L61](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L61) | [L356](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen2/modeling_qwen2.py#L356) |
| final norm and logits | [L71](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L71) | [L465](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen2/modeling_qwen2.py#L465) |
| loss | [L21](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L21) | [L469](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen2/modeling_qwen2.py#L469) |

## Llama — `src/transformers/models/llama/modeling_llama.py` (513 lines)

| place | ours (123 lines) | Llama |
|---|---|---|
| normalization | [L18](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L18) | [L53-L70](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/llama/modeling_llama.py#L53-L70) |
| positions | [L62](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L62) | [L138-L160](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/llama/modeling_llama.py#L138-L160) |
| q, k and v in one matrix | [L29](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L29) | [L217-L281](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/llama/modeling_llama.py#L217-L281) |
| score table `q @ kᵀ` | [L37](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L37) | [L204](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/llama/modeling_llama.py#L204) |
| add the mask | [L38](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L38) | [L206](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/llama/modeling_llama.py#L206) |
| build the mask | — (not present) | [L391](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/llama/modeling_llama.py#L391) |
| softmax | [L39](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L39) | [L208](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/llama/modeling_llama.py#L208) |
| weighted average of v | [L39](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L39) | [L210](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/llama/modeling_llama.py#L210) |
| shared heads (GQA) | — (not present) | [L179-L188](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/llama/modeling_llama.py#L179-L188) |
| MLP | [L50](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L50) | [L163-L176](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/llama/modeling_llama.py#L163-L176) |
| the two residual lines | [L69-L70](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L69-L70) | [L317-L323](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/llama/modeling_llama.py#L317-L323) |
| embedding | [L61](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L61) | [L381](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/llama/modeling_llama.py#L381) |
| final norm and logits | [L71](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L71) | [L480](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/llama/modeling_llama.py#L480) |
| loss | [L21](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L21) | [L484](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/llama/modeling_llama.py#L484) |

## OLMo 2 — `src/transformers/models/olmo2/modeling_olmo2.py` (499 lines)

| place | ours (123 lines) | OLMo 2 |
|---|---|---|
| normalization | [L18](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L18) | [L51-L68](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/olmo2/modeling_olmo2.py#L51-L68) |
| positions | [L62](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L62) | [L162-L185](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/olmo2/modeling_olmo2.py#L162-L185) |
| q, k and v in one matrix | [L29](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L29) | [L196-L266](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/olmo2/modeling_olmo2.py#L196-L266) |
| score table `q @ kᵀ` | [L37](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L37) | [L149](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/olmo2/modeling_olmo2.py#L149) |
| add the mask | [L38](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L38) | [L151](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/olmo2/modeling_olmo2.py#L151) |
| build the mask | — (not present) | [L390](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/olmo2/modeling_olmo2.py#L390) |
| softmax | [L39](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L39) | [L153](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/olmo2/modeling_olmo2.py#L153) |
| weighted average of v | [L39](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L39) | [L155](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/olmo2/modeling_olmo2.py#L155) |
| shared heads (GQA) | — (not present) | [L124-L133](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/olmo2/modeling_olmo2.py#L124-L133) |
| MLP | [L50](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L50) | [L269-L282](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/olmo2/modeling_olmo2.py#L269-L282) |
| the two residual lines | [L69-L70](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L69-L70) | [L316-L322](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/olmo2/modeling_olmo2.py#L316-L322) |
| embedding | [L61](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L61) | [L380](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/olmo2/modeling_olmo2.py#L380) |
| final norm and logits | [L71](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L71) | [L479](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/olmo2/modeling_olmo2.py#L479) |
| loss | [L21](https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py#L21) | [L483](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/olmo2/modeling_olmo2.py#L483) |
