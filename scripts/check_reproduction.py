#!/usr/bin/env python3
"""Answer one question: did you get the same numbers I did?

  python scripts/check_reproduction.py                 run it and compare
  python scripts/check_reproduction.py --log out.txt   compare a run you already have

It runs `OMP_NUM_THREADS=1 python transformer.py 200 input.txt`, compares the
loss column against the reference below, and tells you yes or no. If it says
no, it prints the block to paste into an issue — that report is the single most
useful thing this repository can receive.

The reference was measured on a CPU-only container, torch 2.14.0+cpu,
Python 3.11.2, on 2026-10-16. Timing is not part of it and never will be.
"""
import argparse
import pathlib
import platform
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS_MD5 = "6fb458f1232090904fb40fe944165e91"
CORPUS_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
              "data/tinyshakespeare/input.txt")

# step: (train, val) — exactly as printed, four decimals
REFERENCE = {
  0: ("4.2006", "3.9836"),
  20: ("3.4308", "3.4368"),
  40: ("3.2868", "3.3116"),
  60: ("2.9855", "3.1101"),
  80: ("2.7989", "2.8680"),
  100: ("2.7746", "2.7454"),
  120: ("2.6514", "2.6279"),
  140: ("2.6177", "2.6219"),
  160: ("2.5419", "2.6296"),
  180: ("2.5292", "2.5087"),
  200: ("2.5400", "2.4785"),
}
ROW = re.compile(r"^\s*(\d+)\s+(\d+\.\d{4})\s+(\d+\.\d{4})\s+[\d.]+\s*$", re.M)


def corpus(path):
  """Find input.txt, or say exactly how to get the right one."""
  p = pathlib.Path(path)
  if not p.exists():
    sys.exit(f"{p} is not here. Get the one the reference was measured on:\n"
             f"  curl -O {CORPUS_URL}")
  import hashlib
  got = hashlib.md5(p.read_bytes()).hexdigest()
  if got != CORPUS_MD5:
    sys.exit(f"{p} is not the same file.\n"
             f"  expected md5 {CORPUS_MD5}\n"
             f"  got          {got}\n"
             f"A different corpus changes every number here. Re-download:\n"
             f"  curl -O {CORPUS_URL}")
  return p


def run(corpus_path):
  import os
  env = dict(os.environ, OMP_NUM_THREADS="1")
  cmd = [sys.executable, str(ROOT / "transformer.py"), "200", str(corpus_path)]
  print(f"  $ OMP_NUM_THREADS=1 python transformer.py 200 {corpus_path}\n")
  out = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
  if out.returncode != 0:
    sys.exit(out.stdout + out.stderr)
  return out.stdout


def environment():
  try:
    import torch
    tv = torch.__version__
  except ImportError:
    tv = "not importable"
  return (f"torch      {tv}\n"
          f"python     {platform.python_version()}\n"
          f"platform   {platform.platform()}\n"
          f"machine    {platform.machine()}\n"
          f"processor  {platform.processor() or '(not reported)'}")


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--log", help="compare a run you already captured")
  ap.add_argument("--corpus", default=str(ROOT / "input.txt"))
  args = ap.parse_args()

  text = pathlib.Path(args.log).read_text() if args.log else run(corpus(args.corpus))
  got = {int(s): (t, v) for s, t, v in ROW.findall(text)}

  missing = sorted(set(REFERENCE) - set(got))
  if missing:
    sys.exit(f"the output has no row for step {missing[0]} — is this the output of\n"
             f"`python transformer.py 200 input.txt`?\n\n{text}")

  off = [s for s in sorted(REFERENCE) if got[s] != REFERENCE[s]]
  width = max(len(str(s)) for s in REFERENCE)
  for s in sorted(REFERENCE):
    mark = "  " if s not in off else "<-"
    r, g = REFERENCE[s], got[s]
    line = f"  {s:>{width}}   {g[0]}  {g[1]}"
    print(line if s not in off else f"{line}   {mark} reference {r[0]}  {r[1]}")

  if not off:
    print("\nSAME. All 11 rows match the reference to four decimals.")
    return 0

  print(f"\nDIFFERENT in {len(off)} of {len(REFERENCE)} rows.\n"
        f"This is worth an issue — a reproduction that fails is worth more to\n"
        f"me than a star. Paste the block below into\n"
        f"https://github.com/YoungjuneKwon/ten-line-transformer/issues/new\n")
  print("-" * 68)
  print(environment())
  print()
  print(text.split("\nsample")[0].strip())
  print("-" * 68)
  return 1


if __name__ == "__main__":
  sys.exit(main())
