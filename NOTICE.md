# NOTICE — 제3자 저작물 고지

이 저장소의 코드는 MIT License 로 배포한다(`LICENSE` 참조).
책 원고는 이 저장소에 포함되지 않으며 저작권을 유보한다.

---

## 1. nanoGPT (MIT License)

`transformer.py` 는 **Andrej Karpathy 의 nanoGPT 를 바탕에 두고 다시 쓴 것**이다.
해당 부분은 아래 라이선스의 적용을 받으며, 원문을 그대로 싣는다.

출처: https://github.com/karpathy/nanoGPT (`model.py`)

글자 단위로 비교하면 파일 전체가 1.3%, 어텐션 본문이 22.6% 겹친다. 그런데 **계산의
순서는 열 단계가 같다.** 어디가 어디서 왔는지를 줄 단위로 적는다.

| `transformer.py` | nanoGPT `model.py` | 무엇이 다른가 |
|---|---|---|
| 24–41 `Attention` | `CausalSelfAttention` | **계산 순서가 같다.** 이름이 다르다 (`c_attn`→`qkv`, `c_proj`→`proj`, `bias`→`mask`). dropout 두 곳을 뺐다 |
| 35–36 `qkv(x).split` → `view` → `transpose` | 같은 세 줄 | 형상 주석 `# (B, nh, T, hs)` 까지 같다. 우리는 세 벌을 한 줄의 내포로 묶었다 |
| 37 `(q @ k.transpose(-2,-1)) / math.sqrt(self.hs)` | `… * (1.0 / math.sqrt(k.size(-1)))` | 곱셈 대신 나눗셈으로 썼다 |
| 38 `masked_fill(self.mask[…] == 0, -inf)` | `masked_fill(self.bias[…] == 0, -inf)` | 버퍼 이름만 다르다 |
| 39–41 `softmax @ v` → `transpose().contiguous().view()` → `proj` | 같은 세 줄 | `resid_dropout` 을 뺐다 |
| 43–50 `MLP` | `MLP` | `c_fc`/`c_proj` → `fc`/`out`. 4배 확장과 GELU 는 같다. 편향과 dropout 을 뺐다 |
| 69–70 `x = x + blk.attn(layer_norm(x))` | `x = x + self.attn(self.ln_1(x))` | **문자 그대로 같은 꼴이다.** 우리는 Block 안이 아니라 Transformer.forward 에 두었다 |
| 68 `self.wte[tokens] + self.wpe[:T]` | `tok_emb + pos_emb` | 같은 합이다. 우리는 nn.Embedding 대신 Parameter 를 직접 썼다 |
| 18–19 `layer_norm` | `LayerNorm` 클래스 | **학습되는 배율·이동을 뺐다.** 클래스가 함수 하나가 됐다 |
| 58–72 `Transformer` | `GPT` | `from_pretrained`·`configure_optimizers`·`crop_block_size` 는 가져오지 않았다 |
| 4–16 `load_data`·`get_batch` | `train.py` 의 데이터 준비 | 다시 썼다. nanoGPT 는 `.bin` 을 memmap 으로 읽는다 |
| 74–123 `report`·`train`·`generate`·`cli`·`main` | `train.py`·`sample.py` | 다시 썼다. 두 파일에 흩어진 것을 한 파일에 모았다 |

**`Attention`·`MLP`·그리고 잔차 두 줄의 계보가 가장 짙다.** 그 자리는 2017년 논문에서
이어져 온 계산이라 바꿀 이유가 없었다. 나머지는 읽기 위해 다시 썼다.

전역 변수를 없애고 이름을 형상으로 통일한 판단들은 책 본문에서 다룬다.

```
MIT License

Copyright (c) 2022 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. input.txt — tiny shakespeare

```
크기   1,115,394 bytes
md5    6fb458f1232090904fb40fe944165e91
```

내용은 퍼블릭 도메인인 셰익스피어 텍스트다. 이 특정 파일은 Andrej Karpathy 의
`char-rnn` / `ng-video-lecture` 저장소에서 널리 배포된 것과 바이트 단위로 동일하다.
두 저장소에는 라이선스 파일이 없으나, 퍼블릭 도메인 텍스트를 이어 붙인 것에는
별도의 저작권을 주장하기 어렵다고 판단했다.

**이 파일을 교체하지 않는다.** 교체하면 이 책에 실린 모든 측정 숫자가 달라진다.
독자가 같은 숫자를 얻으려면 같은 파일이어야 한다. 위 md5 가 그 근거다.
