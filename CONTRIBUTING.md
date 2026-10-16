# Contributing

[한국어는 아래](#한국어)

This repository is one file, `transformer.py`, 123 lines. It exists to be read,
not to grow. So the most useful contribution here is not a feature — it is a
correction.

## The one thing I most want to hear

One command does the whole thing — runs it, checks the corpus md5, compares
every row, and prints a ready-to-paste report if anything differs.

```bash
pip install torch
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python scripts/check_reproduction.py
```

It is the same as running this by hand and reading the table yourself:

```bash
OMP_NUM_THREADS=1 python transformer.py 200 input.txt
```

Expected, on any machine, with the seed fixed at 1337:

```
  step     train       val
     0    4.2006    3.9836
   100    2.7746    2.7454
   200    2.5400    2.4785
```

If you see different values, **please open an issue** — there is
[a form for it](https://github.com/YoungjuneKwon/ten-line-transformer/issues/new?template=1-numbers-differ.yml)
that asks for exactly what is needed. A reproduction that fails is worth more
to me than a star.

The same check runs on GitHub's runners on every push and once a week, so the
badge on the front page tells you whether it held somewhere that is not my
machine.

(Timing differs by machine and is not part of the claim. With more than one
thread the addition order changes and the last digit can move — that is why the
command pins threads to one.)

## Other claims you are welcome to falsify

1. **The lineage map.** [NOTICE.md](NOTICE.md) states line by line what came
   from nanoGPT and what changed, and puts the character-level overlap at 1.3%
   for the whole file and 22.6% for the attention body. If a row is wrong,
   incomplete, or understates the overlap, say so. Attribution being correct
   matters more here than it being flattering.
2. **"The order of the computation is the same ten steps."** If the sequence in
   the README does not match what the code actually does, point at the line.
3. **What was left out.** Biases, dropout, learned normalization coefficients
   and weight tying are removed on purpose. If removing one of them makes some
   statement in the README untrue, that is a real bug.
4. **Readability.** The file has no comments by design. If a line stopped you —
   you had to run it to know what it did — tell me which line. That is the
   measurement this repository is actually for.

## Pull requests

Welcome for: a genuine bug, a line that does the same thing in fewer or clearer
lines, or a fix to NOTICE.md / README.

Please do not send: biases, dropout, config classes, flash attention, CLI
frameworks, type annotations, a test directory, or anything that splits the file
into modules. Those are all improvements to a different repository. A patch that
adds a line has to show what the line buys.

Keep the diff small, say what you ran, and paste the output.

## Discussions

Questions about how the code works, or about the transformer itself, belong in
[Discussions](https://github.com/YoungjuneKwon/ten-line-transformer/discussions)
rather than Issues. Issues are for things that are wrong.

---

## 한국어

이 저장소는 `transformer.py` 한 파일, 123줄입니다. 읽히기 위해 있고 자라기 위해
있지 않습니다. 그래서 여기서 가장 쓸모 있는 기여는 기능 추가가 아니라 **정정**입니다.

### 가장 듣고 싶은 것 하나

명령 하나가 다 합니다 — 돌리고, 말뭉치 md5 를 보고, 모든 행을 대조하고,
다르면 그대로 붙일 보고서를 찍어 줍니다.

```bash
python scripts/check_reproduction.py
```

손으로 이렇게 돌리고 표를 직접 읽는 것과 같습니다.

```bash
OMP_NUM_THREADS=1 python transformer.py 200 input.txt
```

시드를 1337로 고정했으니 어느 기계에서도 이 값이 나와야 합니다.

```
  step     train       val
     0    4.2006    3.9836
   100    2.7746    2.7454
   200    2.5400    2.4785
```

다른 값이 나오면 **이슈를 열어 주십시오.** 필요한 것만 묻는
[양식](https://github.com/YoungjuneKwon/ten-line-transformer/issues/new?template=1-numbers-differ.yml)
을 만들어 두었습니다. 재현이 깨졌다는 보고가 별 하나보다 가치 있습니다.

같은 대조가 매 푸시와 매주 한 번 깃헙 러너에서 돕니다. 앞면의 배지가 제 기계가
아닌 곳에서도 성립했는지를 말해 줍니다.

(걸리는 시간은 기기마다 다르고 주장에 들어가지 않습니다. 스레드가 여럿이면 더하는
순서가 달라져 마지막 자리가 움직입니다 — 위 명령이 스레드를 하나로 고정하는 이유입니다.)

### 반증을 환영하는 다른 주장들

1. **계보 대응표.** [NOTICE.md](NOTICE.md) 에 nanoGPT 에서 무엇이 왔고 무엇을
   고쳤는지를 줄 단위로 적었고, 글자 단위 중복을 파일 전체 1.3%, 어텐션 본문
   22.6% 로 적었습니다. 틀린 행, 빠진 행, 중복을 적게 적은 곳이 있으면 지적해
   주십시오. 여기서는 고지가 **유리한 것보다 정확한 것**이 중요합니다.
2. **「계산의 순서는 열 단계가 같다」.** README 의 그 순서가 코드가 실제로 하는
   일과 다르면 줄을 가리켜 주십시오.
3. **빼놓은 것들.** 편향, 드롭아웃, 학습되는 정규화 계수, 가중치 묶기를 일부러
   뺐습니다. 그중 하나를 뺀 탓에 README 의 어떤 서술이 거짓이 된다면 그건 진짜
   버그입니다.
4. **읽힘.** 이 파일에는 주석이 없는 것이 설계입니다. 어느 줄에서 멈추셨다면 —
   돌려 봐야 무슨 일을 하는지 알 수 있었다면 — 그 줄을 알려 주십시오. 이 저장소가
   실제로 재고 싶은 측정이 그것입니다.

### 풀 리퀘스트

환영합니다: 실제 버그, 같은 일을 더 적거나 더 분명한 줄로 하는 변경,
NOTICE.md·README 의 정정.

보내지 마시기를 부탁합니다: 편향, 드롭아웃, 설정 클래스, flash attention, CLI
프레임워크, 타입 주석, 테스트 디렉터리, 파일을 모듈로 쪼개는 변경. 모두 다른
저장소를 좋게 하는 개선입니다. 줄을 늘리는 패치는 그 줄이 무엇을 사는지 보여
주어야 합니다.

diff 를 작게, 무엇을 돌렸는지 적고, 출력을 붙여 주십시오.

### Discussions

코드가 어떻게 돌아가는지, 트랜스포머 자체에 대한 질문은 이슈가 아니라
[Discussions](https://github.com/YoungjuneKwon/ten-line-transformer/discussions)
에 올려 주십시오. 이슈는 **틀린 것**을 위한 자리입니다.
