#!/usr/bin/env python3
"""Generate docs/line-mapping.md from scripts/line-mapping.json.

  python scripts/make_line_mapping.py

Do not edit the markdown by hand — edit the json, or the rows below, and run
this. Every line number in the output comes from the json, which was built by
walking the ast of each file at the pinned tag.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "scripts" / "line-mapping.json").read_text())
DOC = ROOT / "docs" / "line-mapping.md"

OURS_URL = "https://github.com/YoungjuneKwon/ten-line-transformer/blob/main/transformer.py"
HF_URL = "https://github.com/huggingface/transformers/blob/{tag}/{path}"

# (label, anchor key in ours, anchor key in the model file)
# A key of None means the file genuinely has no counterpart — that absence is
# half of what this table is for.
ROWS = [
  ("normalization",            "정규화",        "정규화"),
  ("positions",                "위치",          "위치"),
  ("q, k and v in one matrix", "qkv",           "qkv"),
  ("score table `q @ kᵀ`",     "점수표",        "점수표"),
  ("add the mask",             "마스크",        "마스크"),
  ("build the mask",           None,            "마스크 생성"),
  ("softmax",                  "softmax 와 v",  "softmax"),
  ("weighted average of v",    "softmax 와 v",  "v 가중평균"),
  ("shared heads (GQA)",       None,            "헤드 공유"),
  ("MLP",                      "MLP 순방향",    "MLP"),
  ("the two residual lines",   "잔차 두 줄",    "잔차 두 줄"),
  ("embedding",                "임베딩",        "임베딩"),
  ("final norm and logits",    "로짓",          "로짓"),
  ("loss",                     "손실",          "손실"),
]

TITLES = {"qwen2": "Qwen2.5", "llama": "Llama", "olmo2": "OLMo 2"}


def link(url, span):
  if span is None:
    return "— (not present)"
  lo, hi = span
  if lo is None:
    return "— (not present)"
  frag = f"L{lo}-L{hi}" if hi else f"L{lo}"
  return f"[{frag}]({url}#{frag})"


def main():
  tag = DATA["transformers_tag"]
  ours = DATA["ours"]
  out = [
    "# Line by line, against the official sources",
    "",
    "`scripts/make_line_mapping.py` writes this file. Do not edit it by hand.",
    "",
    "Every row links to a line in a repository that is not mine. The outer",
    "columns are production model code in",
    "[huggingface/transformers](https://github.com/huggingface/transformers),",
    f"pinned at tag `{tag}` so the line numbers keep pointing at the same code.",
    "Click a link and you are looking at the real file.",
    "",
    "**The rows that say *not present* are the point as much as the rest.**",
    f"`transformer.py` is {ours['lines']} lines and a production file is around",
    "500, and this is where the difference goes: building the attention mask,",
    "sharing key and value across heads, rotary positions, caching, config.",
    "None of it changes the order of the computation.",
    "",
    "## Sources",
    "",
    f"- **ours** — [`transformer.py`]({OURS_URL}) · {ours['lines']} lines ·",
    f"  sha256 `{ours['sha16']}`",
    f"- **transformers** — tag `{tag}`, "
    "https://github.com/huggingface/transformers",
    "",
    "If a row is wrong, incomplete, or flatters this file, that is a bug —",
    "please open an issue.",
  ]

  for key, model in DATA["models"].items():
    title = TITLES.get(key, key)
    url = HF_URL.format(tag=tag, path=model["path"])
    out += ["", f"## {title} — `{model['path']}` ({model['lines']} lines)", "",
            f"| place | ours ({ours['lines']} lines) | {title} |", "|---|---|---|"]
    for label, ours_key, model_key in ROWS:
      a = link(OURS_URL, ours["anchors"].get(ours_key) if ours_key else None)
      b = link(url, model["anchors"].get(model_key) if model_key else None)
      out.append(f"| {label} | {a} | {b} |")

  DOC.parent.mkdir(exist_ok=True)
  DOC.write_text("\n".join(out) + "\n")
  print(f"  {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
  main()
