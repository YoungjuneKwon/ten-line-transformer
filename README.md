# ten-line-transformer

[English](README.en.md)

트랜스포머 하나가 파이썬 파일 한 장에 들어 있습니다. **123줄**이고 학습·검증·생성이
다 있습니다. 모듈 수준 변수도, 주석도 없습니다 — 함수 이름이 주석을 대신합니다.

집필 중인 책 『열 줄로 읽는 트랜스포머』(가제)가 이 파일을 처음부터 끝까지 읽습니다.
책 없이 코드만 읽어도 됩니다.

[![Colab에서 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YoungjuneKwon/ten-line-transformer/blob/main/notebook.ipynb)

## 학습 한 스텝은 열 줄입니다

```python
tokens, answers = get_batch()               # (B, T), (B, T)
optimizer.zero_grad(set_to_none=True)
x = wte[tokens] + wpe[:T]                   # (B, T, C)
for blk in blocks:                          # n_block 번
    x = x + blk.attn(layer_norm(x))         # (B, T, C)
    x = x + blk.mlp(layer_norm(x))          # (B, T, C)
logits = layer_norm(x) @ wue                # (B, T, V)
loss = cross_entropy(logits, answers)       # 스칼라 1개
loss.backward()
optimizer.step()
```

실제 파일에서는 가운데 여섯 줄이 `Transformer.forward()` 안에 있고, 나머지 넷이
`train()` 에 있습니다.

## 돌려 보기

브라우저만 있으면 위의 Colab 배지로 바로 돌아갑니다. 내 컴퓨터에서 돌리려면
파이토치와 텍스트 파일 하나가 필요합니다.

```bash
pip install torch
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python transformer.py 200 input.txt
```

CPU 로 돌아갑니다. 그래픽카드는 없어도 됩니다.

## 같은 숫자가 나옵니다

시드를 1337로 고정했습니다. 200 스텝까지 돌리면 손실이 이렇게 내려갑니다.

```
  step     train       val
     0    4.2006    3.9836
   100    2.7746    2.7454
   200    2.5400    2.4785
```

걸리는 시간은 기기마다 다릅니다. 손실 값은 다르지 않습니다.

**숫자가 다르게 나오면 이슈를 열어 주십시오.** 이 저장소가 가장 듣고 싶은 것이고,
무엇을 적어 주시면 되는지는 [CONTRIBUTING.md](CONTRIBUTING.md) 에 있습니다.

## 모델

```
B=32  T=64  C=128  H=4  n_block=4      학습되는 텐서 19장, 숫자 811,264개
wte (65,128)  wpe (64,128)  wue (128,65)
blocks.N.attn.qkv (384,128)  attn.proj (128,128)  mlp.fc (512,128)  mlp.out (128,512)
```

편향, 드롭아웃, 학습되는 정규화 계수, 가중치 묶기는 **일부러 뺐습니다.** 구조를
보이는 데 필요하지 않고, 빼면 읽을 것이 줄어듭니다.

## 어디서 왔나

온라인에 공개된 [nanoGPT](https://github.com/karpathy/nanoGPT)를 바탕에 두고 파이토치로
대대적으로 다시 썼습니다. 글자 단위로 비교하면 파일 전체가 1.3%, 어텐션 본문이 22.6%
겹칩니다. 그런데 **계산의 순서는 열 단계가 같습니다.**

```
qkv 사영 -> 3분할 -> 헤드 분리 -> q·kᵀ -> √ 로 나눔
        -> 마스크 -> softmax -> @ v -> 헤드 병합 -> 출력 사영
```

다시 쓰는 동안 결과가 달라지지 않았는지 매번 확인했습니다. 같은 시드로 200 스텝을
돌리면 학습된 숫자 811,264개가 **한 비트도 다르지 않습니다**(스레드를 하나로
고정했을 때. 여러 스레드로 돌리면 더하는 순서가 달라져 마지막 자리가 흔들립니다).

## 라이선스

MIT. 출처 고지는 [NOTICE.md](NOTICE.md) 에 있습니다.

실험 코드와 측정 로그는 책과 함께 공개합니다.
