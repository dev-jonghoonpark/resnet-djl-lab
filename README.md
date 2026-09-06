# resnet-djl-lab

DJL(Deep Java Library) 블록 API로 **ResNet v1과 v2를 직접 구현**하고 나란히 비교하는 실습 프로젝트.

두 논문의 차이는 새로운 연산이 아니라 **BN·ReLU·덧셈의 순서** 하나다. 그래서 두 구현이
같은 부품(`Layers`)과 같은 스테이지 표(`ResNetSpec`)를 공유하도록 짰다. 차이는
`ResNetV1`과 `ResNetV2` 두 파일에만 남는다.

- v1 — He et al., *Deep Residual Learning for Image Recognition* (CVPR 2016)
- v2 — He et al., *Identity Mappings in Deep Residual Networks* (ECCV 2016)

## 실행

```bash
mvn test          # 16개 검증 테스트
mvn exec:java     # v1/v2 구조 및 파라미터 수 비교 출력

# CIFAR-10 학습
mvn exec:java -Dexec.mainClass=lab.resnet.Train \
  -Dexec.args="--version v2 --depth 20 --epochs 182"
```

Java 21, DJL 0.32.0, PyTorch 엔진. 엔진 네이티브 라이브러리와 CIFAR-10 데이터는 첫
실행 때 `~/.djl.ai`로 자동 내려받는다.

## 무엇이 다른가

### 1. 활성화 위치

```
v1 (post-activation)
  x ──┬─ Conv - BN - ReLU - Conv - BN ─┬─ (+) ─ ReLU ─→
      └────── identity / 1x1 Conv ─────┘

v2 (pre-activation)
  x ──┬─ BN - ReLU - Conv - BN - ReLU - Conv ─┬─ (+) ─→
      └────────────── identity ───────────────┘
```

v1은 덧셈 **뒤에** ReLU가 있다. 항등 경로 위에도 비선형이 얹히므로 입력이 출력까지
그대로 도달하지 못한다. v2는 덧셈이 마지막이라 순수한 항등 경로가 열리고, 역전파
때 그래디언트가 감쇠 없이 흐른다. 1000층이 학습되는 이유가 이것이다.

코드에서는 `ParallelBlock`의 합류 함수 한 줄 차이다.

```java
// ResNetV1
Activation.relu(branch0.add(branch1))
// ResNetV2
branch0.add(branch1)
```

### 2. 투영 지름길의 비대칭 (v2에서 가장 자주 틀리는 곳)

v2에서 채널 수가 바뀔 때는 투영 지름길도 **사전 활성화를 거친 값**을 받는다. 반대로
채널이 그대로일 때는 지름길이 사전 활성화 **이전**의 x를 받는다. 그래서 두 경우의
블록 구조 자체가 다르다 (`ResNetV2.assemble`).

```
채널 동일:  ParallelBlock[ Seq(BN, ReLU, convs), identity ]
채널 변경:  Seq( BN, ReLU, ParallelBlock[ convs, 1x1 Conv ] )
```

투영 conv 뒤에 BN을 붙이지 않는 것도 의도적이다 — 항등 경로를 깨끗하게 두자는 게
v2의 취지다. v1의 투영 지름길에는 BN이 붙는다(논문의 option B). 아래 파라미터 수
차이는 전부 여기서 나온다.

### 3. 스템과 꼬리

|            | v1                              | v2                                       |
|------------|---------------------------------|------------------------------------------|
| CIFAR 스템 | 3x3 Conv - BN - ReLU            | 3x3 Conv (활성화는 첫 유닛이 담당)        |
| 풀링 직전  | 없음                            | **BN - ReLU** (마지막 유닛이 conv로 끝나서) |

## 검증

`mvn exec:java`가 뽑는 파라미터 수는 알려진 값과 정확히 일치한다.

| 모델        | v1 파라미터  | v2 파라미터  | 대조군                    |
|-------------|-------------|-------------|---------------------------|
| ImageNet-18 | 11,689,512  | 11,687,848  | torchvision 11,689,512 ✅ |
| ImageNet-34 | 21,797,672  | 21,796,008  | torchvision 21,797,672 ✅ |
| ImageNet-50 | 25,557,032  | 25,549,480  | torchvision 25,557,032 ✅ |
| CIFAR-20    | 272,474     | 272,282     | 논문 0.27M ✅             |
| CIFAR-56    | 855,770     | 855,578     | 논문 0.85M ✅             |
| CIFAR-164   | 1,704,154   | 1,703,258   | 논문 1.7M ✅              |

`ResNetTest`는 shape 외에 **동작으로 드러나는 차이**도 검증한다.

- `v1UnitOutputIsNeverNegative` — v1 유닛 출력은 덧셈 뒤 ReLU 때문에 절대 음수가 못 된다
- `v2UnitOutputPassesNegativesThrough` — v2는 음수가 그대로 통과한다
- `v2IdentityUnitIsExactIdentityWhenResidualIsZeroed` — 잔차 경로 conv 가중치를 0으로
  만들면 출력이 입력과 **비트 단위로** 같아진다. 항등 사상이 실제로 성립한다는 직접 증거

## 학습 (CIFAR-10)

`Train`은 논문 4.2절의 설정을 그대로 쓴다 — SGD momentum 0.9, weight decay 1e-4,
초기 lr 0.1, 전체 업데이트의 50%·75% 지점에서 lr을 1/10로, MSRA/He 초기화,
4픽셀 패딩 후 랜덤 크롭 + 좌우 반전 증강.

```bash
# 빠른 확인 (2~3분)
mvn exec:java -Dexec.mainClass=lab.resnet.Train \
  -Dexec.args="--version v1 --depth 20 --epochs 3 --limit 5120"

# 논문 스케줄
mvn exec:java -Dexec.mainClass=lab.resnet.Train \
  -Dexec.args="--version v2 --depth 110 --epochs 182"
```

| 옵션            | 기본값 | 설명                                        |
|-----------------|--------|---------------------------------------------|
| `--version`     | `v1`   | `v1` 또는 `v2`                              |
| `--depth`       | `20`   | 6n+2 (basic) 또는 9n+2 (bottleneck)         |
| `--epochs`      | `2`    | 논문 스케줄은 182                            |
| `--batch`       | `128`  |                                             |
| `--lr`          | `0.1`  | 초기 학습률                                  |
| `--limit`       | 없음   | 학습 샘플 수 제한 — 스모크 런용              |
| `--no-augment`  | -      | 증강 끄기 (증강 효과 확인용)                 |
| `--max-gpus`    | `1`    | GPU가 없으면 자동으로 CPU                    |
| `--out`         | `build/model` | 체크포인트 저장 경로                  |

에폭마다 검증하고 체크포인트를 남긴다.

### 실측 (Apple Silicon CPU, PyTorch 엔진)

| 실행                          | 결과                    | 소요   |
|-------------------------------|-------------------------|--------|
| v1-20, 1 에폭, 전체 50k       | 검증 정확도 **53.07%**  | 5분    |
| v2-20, 3 에폭, 5,120 샘플     | 검증 정확도 **34.90%**  | 2분    |

**CPU로 논문 스케줄을 다 돌리면 약 15시간이다** (182 에폭 × 5분). 제대로 재현하려면
GPU를 쓰거나 에폭 수를 줄이는 편이 낫다. 논문 기준 ResNet-20의 최종 오류율은 8.75%다.

## 사이클 한 번 들여다보기 (`Probe`)

이미지 **한 장**이 순전파 → 손실 → 역전파 → 가중치 갱신 → 재순전파를 한 번 도는 동안
오간 값을 전부 받아 적어 JSON으로 남긴다. 그 JSON을 HTML 한 장으로 말아 넣으면
단계별로 값이 어떻게 변하는지 눈으로 볼 수 있다.

**결과물 → https://dev-jonghoonpark.github.io/resnet-djl-lab/**

```bash
mvn exec:java -Dexec.mainClass=lab.resnet.Probe   # build/probe/trace-{v1,v2}.json
python3 tools/viz/build.py                        # docs/index.html + build/viz/cycle.html
```

`build.py`는 두 벌을 뽑는다. `docs/index.html`은 GitHub Pages가 그대로 서빙하는 완전한
문서이고, `build/viz/cycle.html`은 `<head>`를 스스로 씌우는 뷰어에 넣을 조각이다. 둘 다
외부 요청이 하나도 없다 — trace JSON도, 이미지도, 히트맵도 전부 인라인이다.

`docs/`가 바뀐 채로 `main` 또는 `feat/resnet-v1-v2`에 push하면
`.github/workflows/pages.yml`이 그대로 올린다. CI는 Java를 돌리지 않는다 — HTML이 이미
커밋되어 있기 때문이다. 그래서 `Probe`를 다시 돌리는 것은 사람의 몫이다.

| 옵션        | 기본값                                  | 설명                          |
|-------------|-----------------------------------------|-------------------------------|
| `--version` | `v1,v2`                                 | 쉼표로 여러 개                |
| `--depth`   | `20`                                    |                               |
| `--image`   | `assets/cifar10/test/cat/test_01124.png`| 상위 폴더명이 정답 클래스     |
| `--lr`      | `0.1`                                   | 갱신 스텝의 학습률            |
| `--seed`    | `42`                                    | 초기화 시드                   |
| `--out`     | `build/probe`                           |                               |

기록하는 것:

- 전처리 3단계(PNG uint8 → ToTensor → Normalize)에서 픽셀 하나가 겪는 값 변화
- 블록 하나하나의 입출력 shape · min/max/평균/표준편차 · 0의 비율 · MAC 수 · 특징 맵
- **연산 한 칸의 산수 전개** — conv 곱셈 27개, BN의 μ·σ²·γ·β, 잔차 덧셈, 평균 풀링 64칸,
  Linear 내적 64항. 전부 Java에서 손으로 다시 계산해 엔진 값과 나란히 둔다
- softmax · 손실, `dL/dz = softmax(z) − onehot`, 마지막 Linear 가중치 그래디언트 대조
- 층별 그래디언트 노름, SGD 한 스텝 전후의 가중치, 갱신 후 다시 넣었을 때의 로짓

`Train`과 다른 점은 배치가 1이라는 것 하나다. 그래서 **BatchNorm의 평균·분산이 그 한 장에서만
나온다**. 초기화·손실·옵티마이저 설정은 `Train`과 같다.

블록 트리를 직접 걸어 내려가며 자식마다 `forward`를 따로 호출한다 — `SequentialBlock.forward`에
맡기면 중간값을 볼 수 없기 때문이다. `ParallelBlock`의 합류 함수는 private이라 꺼낼 수 없어서
버전별 규칙(v1은 더한 뒤 ReLU, v2는 그냥 더하기)을 `Probe`에서 다시 적용한다. 즉 `ResNetV1`/
`ResNetV2`의 합류 함수를 고치면 `Probe.walk`도 같이 고쳐야 한다.

## 파일

| 파일                    | 역할                                                       |
|-------------------------|------------------------------------------------------------|
| `Layers.java`           | conv/BN 부품. v1·v2가 공유 — 차이가 부품이 아님을 드러낸다  |
| `ResNetSpec.java`       | 깊이 → 스테이지 구성 표. v1·v2가 동일하게 쓴다              |
| `ResNetV1.java`         | post-activation 잔차 유닛 + 네트워크 조립                   |
| `ResNetV2.java`         | pre-activation 잔차 유닛 + 네트워크 조립                    |
| `Main.java`             | 구조 덤프와 파라미터 수 비교                                |
| `Train.java`            | CIFAR-10 학습 루프                                          |
| `CifarAugmentation.java`| 패딩 + 랜덤 크롭 + 좌우 반전                                |
| `Probe.java`            | 이미지 한 장의 사이클 한 번을 값 단위로 기록 → JSON         |
| `tools/viz/`            | 그 JSON을 자체 완결 HTML 한 장으로 만드는 템플릿과 빌드 스크립트 |

## DJL 기본 모델주(model-zoo)를 쓰지 않은 이유

`ai.djl:model-zoo`에 `ai.djl.basicmodelzoo.cv.classification.ResNetV1`이 있지만
직접 짜는 쪽을 택했다. v2가 아예 없다는 점 외에도:

1. **ResNet-18/34가 논문 구조가 아니다.** `ResNetV1.Builder.build()`는 224x224 입력에서
   `numLayers < 50`일 때도 `bottleneck = true`로 두고 filters를 `{64, 64, 128, 256, 512}`로
   잡는다. 그 결과 첫 스테이지가 1x1(16) - 3x3(16) - 1x1(64)로 만들어진다. 논문의
   ResNet-18은 3x3 - 3x3 basic 블록이다.
2. **첫 유닛이 항상 투영 지름길을 쓴다.** 채널과 해상도가 그대로여도 1x1 conv + BN을
   태운다. ResNet-18의 첫 스테이지에는 원래 투영이 없다. 이 프로젝트는
   `inChannels == filters && stride == 1`을 직접 판정한다.
3. **스템 분기 기준이 어긋난다.** `Builder.build()`는 `height <= 28`로 CIFAR 구성을
   고르는데 `resnet()`은 `height <= 32`로 스템을 고른다. 32x32 입력이면 ImageNet용
   4-스테이지 구성에 CIFAR용 3x3 스템이 붙는다.

## 알아둘 것

- **병목 유닛의 stride 위치**: 원논문대로 첫 1x1에 걸었다. torchvision의 이른바 "v1.5"는
  가운데 3x3으로 옮겨 정확도를 조금 올린다.
- **v2의 ImageNet 스템 중복**: 스템이 BN-ReLU-maxpool로 끝나고 첫 유닛이 다시 BN-ReLU로
  시작한다. 중복이지만 He et al.의 원 구현이 그렇게 되어 있어 그대로 뒀다.
- **증강의 크롭 오프셋은 배치가 공유한다**. DJL의 `ArrayDataset`은 `BulkDataIterable`을
  쓰기 때문에 파이프라인이 샘플이 아니라 배치 전체에 적용된다. 좌우 반전은 마스크와
  `where`로 샘플마다 따로 걸었지만, 크롭 오프셋 뽑기는 벡터화가 까다로워 배치당 하나로
  뒀다. 매 에폭 셔플되므로 한 이미지는 여러 에폭에 걸쳐 다양한 크롭을 겪는다.
- 학습은 CIFAR-10만 지원한다. ImageNet 스펙은 구조 정의와 shape 검증까지가 범위다.
