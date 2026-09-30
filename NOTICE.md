# NOTICE — 제3자 저작물 고지

이 저장소의 코드는 MIT License 로 배포한다(`LICENSE` 참조).
책 원고는 이 저장소에 포함되지 않으며 저작권을 유보한다.

---

## 1. nanoGPT (MIT License)

`transformer.py` 는 **Andrej Karpathy 의 nanoGPT 를 바탕에 두고 다시 쓴 것**이다.
해당 부분은 아래 라이선스의 적용을 받으며, 원문을 그대로 싣는다.

출처: https://github.com/karpathy/nanoGPT (`model.py`)

줄 단위 대응은 책에서 다룬다. 글자 단위로 비교하면 파일 전체가 1.3%,
어텐션 본문이 22.6% 겹친다 — 계산의 순서는 열 단계가 같다.

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
