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
