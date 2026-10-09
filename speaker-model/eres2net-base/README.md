# ERes2Net (한국어 화자 구분용)

웹판이 한국어 녹음의 화자를 가를 때 쓰는 목소리 모델이다. `diarize.js` 가 이 폴더를 읽는다.

- 원본: [iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k](https://www.modelscope.cn/models/iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k) (3D-Speaker, Alibaba DAMO)
- ONNX 변환본: [sherpa-onnx 화자 인식 모델](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models) 의 `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx`
- 라이선스: Apache License 2.0 (원본과 같음)
- 바꾼 점: transformers.js 가 읽도록 입력 이름만 `x` → `input_features` 로 바꿨다. 가중치와 계산은 그대로다
  (같은 입력에 같은 출력이 나오는 것을 확인했다). `config.json` 은 transformers.js 용으로 새로 썼다.
- `onnx/model.onnx` sha256: `73707f36f778b221ee0aaaa275517fc4e72e37af3537d5c366b39ebdf5d0501c`

GitHub 릴리스 파일은 브라우저가 직접 받지 못해(CORS) 사이트와 같은 저장소에 넣었다.
Git LFS 로 바꾸면 안 된다 — GitHub Pages 는 LFS 파일을 내주지 않는다.
