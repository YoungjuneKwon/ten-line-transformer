#!/usr/bin/env python3
"""Draw the two README figures, once per language.

  python scripts/make_figures.py              all figures, all languages
  python scripts/make_figures.py flow         one figure, all languages
  python scripts/make_figures.py flow en      one figure, one language

Output goes to figures/<name>.<lang>.png. Nothing here is hand-placed data:
the code map is drawn from scripts/line-mapping.json, which was produced by
walking the ast of each file. If a file changes, regenerate the json first.

The figures are for a screen, not for print. Colour carries meaning here.

Needs matplotlib. The three CJK languages need Noto Sans CJK — on Debian that
is `fonts-noto-cjk`. Without it those three fall back and the glyphs come out
as boxes; the English figure is unaffected.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
MAPPING = json.loads((ROOT / "scripts" / "line-mapping.json").read_text())

for ttc in ("NotoSansCJK-Regular.ttc", "NotoSansCJK-Bold.ttc"):
  path = f"/usr/share/fonts/opentype/noto/{ttc}"
  if pathlib.Path(path).exists():
    font_manager.fontManager.addfont(path)

FONTS = {
  "en": ("DejaVu Sans", "DejaVu Sans Mono"),
  "ko": ("Noto Sans CJK KR", "Noto Sans Mono CJK KR"),
  "ja": ("Noto Sans CJK JP", "Noto Sans Mono CJK JP"),
  "zh": ("Noto Sans CJK SC", "Noto Sans Mono CJK SC"),
}
LANGS = list(FONTS)

INK = "#1b1b1b"
MUTED = "#575757"
FAINT = "#9a9a9a"
RULE = "#d6d6d2"
FILL = "#f4f4f2"
HUES = ["#2f6f9f", "#8a5a1f", "#3f7a4a", "#7a3f6f",
        "#1f6f7a", "#9f4a3a", "#5a5a8f", "#2f5f8f"]

SANS, MONO = FONTS["en"]


def use(lang):
  global SANS, MONO
  SANS, MONO = FONTS[lang]
  plt.rcParams.update({"font.family": SANS, "font.size": 9,
                       "figure.dpi": 200, "savefig.dpi": 200,
                       "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
                       "savefig.facecolor": "white"})


def save(fig, name, lang):
  OUT.mkdir(exist_ok=True)
  # The social card has to come out at exactly 1280x640, so it keeps the
  # canvas it was given instead of being cropped to its ink.
  exact = {"bbox_inches": None, "pad_inches": 0} if name == "social" else {}
  fig.savefig(OUT / f"{name}.{lang}.png", **exact)
  plt.close(fig)
  print(f"  figures/{name}.{lang}.png")


# ----------------------------------------------------------------- strings

T = {
  "flow.text":    {"en": "text", "ko": "텍스트", "ja": "テキスト", "zh": "文本"},
  "flow.ids":     {"en": "token ids", "ko": "번호 배열",
                   "ja": "番号の列", "zh": "编号序列"},
  "flow.update":  {"en": "parameters move", "ko": "파라미터 갱신",
                   "ja": "パラメータ更新", "zh": "参数更新"},
  "flow.scalar":  {"en": "one scalar", "ko": "스칼라 1개",
                   "ja": "スカラー 1 個", "zh": "一个标量"},
  "flow.inside":  {"en": "the numbers inside the model", "ko": "모델 안의 숫자들",
                   "ja": "モデルの中の数", "zh": "模型里的数"},
  "flow.tok":     {"en": "tokenize", "ko": "토크나이즈",
                   "ja": "トークナイズ", "zh": "分词"},
  "flow.batch":   {"en": "get_batch — draw B sequences",
                   "ko": "get_batch — 묶음 B개를 뽑는다",
                   "ja": "get_batch — B 本を取り出す",
                   "zh": "get_batch — 抽出 B 条"},
  "flow.embed":   {"en": "embed — wte[tokens] + wpe[:T]",
                   "ko": "임베딩 — wte[tokens] + wpe[:T]",
                   "ja": "埋め込み — wte[tokens] + wpe[:T]",
                   "zh": "嵌入 — wte[tokens] + wpe[:T]"},
  "flow.blocks":  {"en": "n_block blocks — attention and MLP",
                   "ko": "블록 n_block 개 — 어텐션과 MLP",
                   "ja": "ブロック n_block 個 — アテンションと MLP",
                   "zh": "n_block 个块 — 注意力和 MLP"},
  "flow.emph":    {"en": "attention and MLP", "ko": "어텐션과 MLP",
                   "ja": "アテンションと MLP", "zh": "注意力和 MLP"},
  "flow.once":    {"en": "once — setup", "ko": "한 번만 — 준비",
                   "ja": "一度だけ — 準備", "zh": "只做一次 — 准备"},
  "flow.every":   {"en": "every step — ten lines", "ko": "매 스텝 — 열 줄",
                   "ja": "毎ステップ — 十行", "zh": "每一步 — 十行"},

  "map.lines":    {"en": "lines", "ko": "줄", "ja": "行", "zh": "行"},
  "map.solid":    {"en": "solid — does the same job",
                   "ko": "실선 — 하는 일이 같다",
                   "ja": "実線 — 同じ仕事をする",
                   "zh": "实线 — 做同一件事"},
  "map.dashed":   {"en": "dashed — same place, a different part",
                   "ko": "점선 — 자리는 같고 부품이 바뀌었다",
                   "ja": "点線 — 位置は同じ、部品が違う",
                   "zh": "虚线 — 位置相同，零件不同"},
  "map.grey":     {"en": "grey — no counterpart",
                   "ko": "회색 — 대응되지 않는 기호",
                   "ja": "灰色 — 対応しない記号",
                   "zh": "灰色 — 没有对应的符号"},
  "map.matched":  {"en": "matched", "ko": "대응되는 기호",
                   "ja": "対応する記号", "zh": "对应的符号"},
  "map.symbols":  {"en": "symbols", "ko": "개 기호",
                   "ja": "個の記号", "zh": "个符号"},
  "map.note":     {"en": "Each file is laid out whole, top to bottom. Column heights are "
                         "matched, so a line is a different height in each column.\n"
                         "Symbols were taken exhaustively with ast — this is not a selection.",
                   "ko": "파일 전체를 세로로 폈다. 세로 길이를 맞춰 그렸으므로 줄당 높이는 열마다 다르다.\n"
                         "기호는 ast 로 전수를 뽑았다 — 발췌가 아니다.",
                   "ja": "ファイル全体を縦に伸ばした。縦の長さを揃えたので、1 行の高さは列ごとに違う。\n"
                         "記号は ast で全数を取り出した — 抜粋ではない。",
                   "zh": "把整个文件纵向展开。各列高度取齐，所以每行的高度逐列不同。\n"
                         "符号用 ast 全量提取 — 不是摘录。"},
}


def t(key, lang):
  return T[key][lang]


# -------------------------------------------------------------- figure one

FLOW = [
  # (name key or literal, shape key or literal, arrow key, group)
  ("flow.text",   '"First Citizen:"', "flow.tok",    None),
  ("flow.ids",    "[18, 47, 56, …]",  "flow.batch",  None),
  ("tokens",      "(B, T)",           "flow.embed",  1),
  ("x",           "(B, T, C)",        "flow.blocks", 2),
  ("x",           "(B, T, C)",        "@ wue",       3),
  ("logits",      "(B, T, V)",        "cross_entropy(answers)", 4),
  ("loss",        "flow.scalar",      "backward · step",        4),
  ("flow.update", "flow.inside",      None,          4),
]
STRONG = {"tokens", "x", "logits"}
SPLIT = 2          # rows from here down are the ten lines


def flow(lang):
  W, H = 100.0, 74.0
  fig, ax = plt.subplots(figsize=(9.6, 9.6 * H / W))
  ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

  bh, gap, split_gap = 5.0, 3.6, 7.0
  wb, xb = 26.0, 11.0
  xshape = W - 1.0
  rows = len(FLOW)
  gaps = [split_gap if i == SPLIT - 1 else gap for i in range(rows - 1)]

  ys, y = [], H - 1.5
  for i in range(rows):
    y -= bh
    ys.append(y)
    if i < rows - 1:
      y -= gaps[i]

  for i, (name_k, shape_k, edge_k, _g) in enumerate(FLOW):
    y = ys[i]
    name = t(name_k, lang) if name_k in T else name_k
    shape = t(shape_k, lang) if shape_k in T else shape_k
    strong = name_k in STRONG
    ax.add_patch(FancyBboxPatch((xb, y), wb, bh, zorder=2, lw=0.9,
                                boxstyle="round,pad=0,rounding_size=1.0",
                                fc=FILL if strong else "white", ec=RULE))
    ax.text(xb + wb / 2, y + bh / 2, name, ha="center", va="center", zorder=3,
            fontsize=11 if strong else 10, color=INK,
            fontweight="bold" if strong else "normal",
            fontfamily=MONO if strong else SANS)
    label = ax.text(xshape, y + bh / 2, shape, ha="right", va="center", fontsize=10,
                    color=INK if strong else MUTED,
                    fontfamily=MONO if shape_k not in T else SANS)
    # The leader has to stop where the text actually starts. Proportional faces
    # make character counting useless, so measure the drawn label instead.
    fig.canvas.draw()
    bb = label.get_window_extent(renderer=fig.canvas.get_renderer())
    xstart = ax.transData.inverted().transform((bb.x0, bb.y0))[0]
    ax.plot([xb + wb + 1.6, xstart - 1.6], [y + bh / 2] * 2,
            color=FAINT, lw=0.4, ls=(0, (1, 2.4)), zorder=0)

    if edge_k is None:
      continue
    edge = t(edge_k, lang) if edge_k in T else edge_k
    xm = xb + wb / 2
    ax.add_patch(FancyArrowPatch((xm, y - 0.4), (xm, y - gaps[i] + 0.5),
                                 arrowstyle="-|>", mutation_scale=9,
                                 lw=1.2, color=INK, shrinkA=0, shrinkB=0, zorder=1))
    emph = t("flow.emph", lang)
    pre, hit, _ = edge.partition(emph)
    lab = ax.text(xm + 2.2, y - gaps[i] / 2, pre if hit else edge, ha="left",
                  va="center", fontsize=9.6, color=MUTED, fontfamily=SANS)
    if hit:
      fig.canvas.draw()
      bb = lab.get_window_extent(renderer=fig.canvas.get_renderer())
      xend = ax.transData.inverted().transform((bb.x1, bb.y0))[0]
      ax.text(xend + 0.5, y - gaps[i] / 2, hit, ha="left", va="center",
              fontsize=10, color=INK, fontweight="bold", fontfamily=SANS)

  # the two clocks. get_batch crossing this line is itself the answer:
  # the array is built once and drawn from on every step.
  ysplit = ys[SPLIT - 1] - 1.7
  ax.plot([0.8, W - 0.8], [ysplit] * 2, color=RULE, lw=0.8,
          ls=(0, (3.5, 2.6)), zorder=0)
  ax.text(xshape, ysplit + 1.1, t("flow.once", lang), ha="right", va="bottom",
          fontsize=9, color=MUTED, fontfamily=SANS)
  ax.text(xshape, ysplit - 1.1, t("flow.every", lang), ha="right", va="top",
          fontsize=9, color=INK, fontweight="bold", fontfamily=SANS)

  for g, mark in zip((1, 2, 3, 4), ("①", "②", "③", "④")):
    idx = [i for i, s in enumerate(FLOW) if s[3] == g]
    ytop = ysplit - 2.6 if g == 1 else ys[idx[0] - 1] - 2.6
    ybot = ys[idx[-1]] + 0.4
    ax.plot([xb - 4.0] * 2, [ybot, ytop], color=HUES[g - 1], lw=1.8,
            solid_capstyle="round", zorder=1, alpha=0.75)
    ax.text(xb - 4.0, ytop + 0.4, mark, ha="center", va="bottom",
            fontsize=9.5, color=HUES[g - 1], fontfamily=SANS)
  return fig


# -------------------------------------------------------------- figure two

GROUPS = [
  # (ours anchor, model anchor, same job?)
  ("임베딩",       "임베딩",       True),
  ("위치",         "위치",         False),
  ("정규화",       "정규화",       False),
  ("qkv",          "qkv",          False),
  ("점수표",       "점수표",       True),
  ("softmax 와 v", "softmax",      True),
  ("MLP 순방향",   "MLP",          False),
  ("잔차 두 줄",   "잔차 두 줄",   True),
]


def owner(symbols, line):
  for k, sm in enumerate(symbols):
    if sm["start"] <= line <= sm["end"]:
      return k
  return None


def codemap(lang):
  cols = [("Qwen2.5", "model", MAPPING["models"]["qwen2"]),
          ("ten-line-transformer", "ours", MAPPING["ours"]),
          ("OLMo 2", "model", MAPPING["models"]["olmo2"])]

  fig, ax = plt.subplots(figsize=(10.4, 12.2))
  ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

  cw, gap = 23.0, 12.5
  xs = [2.0, 2.0 + cw + gap, 2.0 + 2 * (cw + gap)]
  ytop, ybot = 90.0, 16.0
  span = ytop - ybot

  placed = []
  for (title, kind, d), x in zip(cols, xs):
    n = d["lines"]
    ax.text(x + cw / 2, ytop + 4.4, title, ha="center", va="bottom",
            fontsize=10.5, color=INK, fontfamily=SANS, fontweight="bold")
    stem = d["path"].split("/")[-1] if kind == "model" else "transformer.py"
    ax.text(x + cw / 2, ytop + 2.0, f"{stem} · {n} {t('map.lines', lang)}",
            ha="center", va="bottom", fontsize=8, color=MUTED, fontfamily=MONO)

    marked = {}
    for gi, (ours_key, model_key, _same) in enumerate(GROUPS):
      anchor = d["anchors"].get(ours_key if kind == "ours" else model_key, [None])[0]
      if anchor is None:
        continue
      w = owner(d["symbols"], anchor)
      if w is not None:
        marked.setdefault(w, []).append(gi)

    pos = {}
    for k, sm in enumerate(d["symbols"]):
      y1 = ytop - (sm["start"] - 1) / n * span
      y0 = ytop - sm["end"] / n * span
      h = y1 - y0
      if sm["kind"] not in ("def", "class"):
        ax.add_patch(FancyBboxPatch((x, y0), cw, max(h, 0.18),
                                    boxstyle="square,pad=0", fc="#ececea",
                                    ec="none", zorder=1))
        continue
      hit = k in marked
      col = HUES[marked[k][0]] if hit else FAINT
      ax.add_patch(FancyBboxPatch((x, y0), cw, max(h, 0.5),
                                  boxstyle="square,pad=0",
                                  fc=FILL if hit else "white", ec=col,
                                  lw=1.3 if hit else 0.6, zorder=2))
      if h >= (0.95 if hit else 1.25):
        ax.text(x + 0.8, (y0 + y1) / 2, sm["name"][:23], ha="left", va="center",
                fontsize=6.4 if hit else 5.9, color=INK if hit else MUTED,
                fontfamily=MONO, fontweight="bold" if hit else "normal", zorder=3)
      pos[k] = (y0, y1)
    placed.append((x, pos, marked))

  for (xa, pa, ma), (xb_, pb, mb) in zip(placed[:-1], placed[1:]):
    drawn = set()
    for ka, gis in ma.items():
      for gi in gis:
        kb = next((k for k, g in mb.items() if gi in g), None)
        if kb is None or (ka, kb) in drawn or ka not in pa or kb not in pb:
          continue
        drawn.add((ka, kb))
        same = GROUPS[gi][2]
        ax.plot([xa + cw + 0.4, xb_ - 0.4], [sum(pa[ka]) / 2, sum(pb[kb]) / 2],
                color=HUES[gi], alpha=0.95 if same else 0.72,
                lw=1.1 if same else 0.9, ls="-" if same else (0, (2.4, 2.0)),
                zorder=1, solid_capstyle="round", dash_capstyle="round")

  y = ybot - 3.4
  ax.plot([2.0, 96.0], [y + 1.8, y + 1.8], lw=0.7, color=FAINT)
  ax.text(2.0, y, "      ".join([t("map.solid", lang), t("map.dashed", lang),
                                 t("map.grey", lang)]),
          ha="left", va="top", fontsize=8, color=MUTED, fontfamily=SANS)

  ax.text(2.0, y - 3.6, t("map.matched", lang), ha="left", va="top",
          fontsize=8, color=MUTED, fontfamily=SANS)
  for (title, _kind, d), (_x, _pos, marked), dy in zip(cols, placed, (3.6, 6.4, 9.2)):
    named = [s for s in d["symbols"] if s["kind"] in ("def", "class")]
    cov = sum(d["symbols"][k]["end"] - d["symbols"][k]["start"] + 1 for k in marked)
    ax.text(19.0, y - dy, f"{title}   {len(marked)}/{len(named)} {t('map.symbols', lang)}"
                          f" · {cov}/{d['lines']} {t('map.lines', lang)}",
            ha="left", va="top", fontsize=8, color=INK, fontfamily=SANS)

  ax.text(2.0, y - 13.4, t("map.note", lang), ha="left", va="top",
          fontsize=7.4, color=MUTED, fontfamily=SANS, linespacing=1.6)
  ax.text(2.0, y - 19.6,
          f"transformers {MAPPING['transformers_tag']} · "
          f"transformer.py sha256 {MAPPING['ours']['sha16']}",
          ha="left", va="top", fontsize=7, color=FAINT, fontfamily=MONO)
  return fig


# ------------------------------------------------------ the social preview

def social(lang):
  """1280x640 for GitHub's social preview — the card every shared link shows.

  GitHub has no API for setting it; upload this by hand under
  Settings -> General -> Social preview. English only: the card is seen by
  whoever shares the link, not by whoever reads the README.
  """
  cols = [("Qwen2.5", "modeling_qwen2.py", MAPPING["models"]["qwen2"]["lines"]),
          ("ten-line-transformer", "transformer.py", MAPPING["ours"]["lines"]),
          ("OLMo 2", "modeling_olmo2.py", MAPPING["models"]["olmo2"]["lines"])]

  fig, ax = plt.subplots(figsize=(6.4, 3.2))
  ax.set_xlim(0, 128); ax.set_ylim(0, 64); ax.axis("off")
  fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

  ax.text(9, 55.5, "A transformer in one file.", ha="left", va="center",
          fontsize=17, color=INK, fontweight="bold", fontfamily=SANS)
  ax.text(9, 48.0, "123 lines — and one training step is ten of them.",
          ha="left", va="center", fontsize=11, color=MUTED, fontfamily=SANS)

  # Bars in proportion to file length. At thumbnail size the heights are the
  # whole message; the numbers are for anyone who looks twice.
  tallest = max(n for _, _, n in cols)
  base, span = 17.0, 24.0
  bw, gap = 22.0, 11.0
  x = (128 - (3 * bw + 2 * gap)) / 2
  for (title, stem, n), hue in zip(cols, (HUES[0], HUES[3], HUES[2])):
    h = span * n / tallest
    ours = stem == "transformer.py"
    ax.add_patch(FancyBboxPatch((x, base), bw, h, boxstyle="square,pad=0",
                                fc=hue if ours else "#e8e8e6",
                                ec=hue if ours else RULE, lw=1.2, zorder=2))
    ax.text(x + bw / 2, base + h + 2.8, str(n), ha="center", va="center",
            fontsize=15 if ours else 12.5, fontfamily=MONO,
            fontweight="bold" if ours else "normal",
            color=hue if ours else MUTED)
    ax.text(x + bw / 2, 13.2, title, ha="center", va="center",
            fontsize=8.5 if ours else 8, fontfamily=SANS,
            fontweight="bold" if ours else "normal",
            color=INK if ours else MUTED)
    ax.text(x + bw / 2, 9.8, stem, ha="center", va="center",
            fontsize=6.5, fontfamily=MONO, color=FAINT)
    x += bw + gap

  ax.plot([9, 119], [6.6, 6.6], lw=0.8, color=RULE)
  ax.text(9, 3.4, "Same places, mapped line by line.", ha="left", va="center",
          fontsize=8, color=MUTED, fontfamily=SANS)
  ax.text(119, 3.4, "github.com/YoungjuneKwon/ten-line-transformer",
          ha="right", va="center", fontsize=8, color=INK, fontfamily=MONO)
  return fig


FIGURES = {"flow": flow, "codemap": codemap, "social": social}
SINGLE_LANGUAGE = {"social": "en"}      # the card is not per-reader


def main(argv):
  names = [a for a in argv if a in FIGURES] or list(FIGURES)
  langs = [a for a in argv if a in LANGS] or LANGS
  unknown = [a for a in argv if a not in FIGURES and a not in LANGS]
  if unknown:
    sys.exit(f"unknown: {' '.join(unknown)}\n"
             f"figures: {' '.join(FIGURES)}\nlanguages: {' '.join(LANGS)}")
  done = set()
  for lang in langs:
    use(lang)
    for name in names:
      only = SINGLE_LANGUAGE.get(name)
      if only and (name in done or (only not in langs and lang != langs[0])):
        continue
      if only:
        if only in langs and lang != only:
          continue
        use(only)
        save(FIGURES[name](only), name, only)
        use(lang)
        done.add(name)
        continue
      save(FIGURES[name](lang), name, lang)


if __name__ == "__main__":
  main(sys.argv[1:])
