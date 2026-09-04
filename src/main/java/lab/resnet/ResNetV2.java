package lab.resnet;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

/**
 * ResNet v2 — He et al., "Identity Mappings in Deep Residual Networks" (ECCV 2016).
 *
 * <p>핵심은 <b>pre-activation</b>이다. BN과 ReLU가 conv <em>앞으로</em> 옮겨가고, 덧셈
 * 뒤에는 아무것도 붙지 않는다. 그 결과 입력에서 출력까지 어떤 변형도 거치지 않는
 * 순수한 항등 경로가 생기고, 역전파 때 그래디언트가 그 길로 감쇠 없이 흐른다.
 * 1000층짜리 네트워크가 학습되는 이유가 이것이다.
 *
 * <pre>
 *   x ──┬─ BN - ReLU - Conv - BN - ReLU - Conv ─┬─ (+) ─→   (채널이 그대로일 때)
 *       └──────────── identity ─────────────────┘
 *
 *   x ─ BN - ReLU ─┬─ Conv - BN - ReLU - Conv ─┬─ (+) ─→   (채널이 바뀔 때)
 *                  └────── 1x1 Conv ───────────┘
 * </pre>
 *
 * <p>아래쪽 형태를 눈여겨볼 것. 채널이 바뀔 때는 투영 지름길도 사전 활성화를 거친
 * 값을 받는다 (원 구현의 {@code shortcut = Conv(act1, ...)}). 반대로 채널이 그대로일
 * 때는 지름길이 사전 활성화 <em>이전</em>의 x를 받는다. 이 비대칭이 v2를 v1처럼
 * 짜다가 가장 자주 틀리는 지점이다.
 */
public final class ResNetV2 {

    /** 그냥 더하기. v1과 달리 여기에 ReLU가 없다 — 그것이 v2의 전부다. */
    private static final Function<List<NDList>, NDList> ADD =
            branches ->
                    new NDList(
                            branches.get(0)
                                    .singletonOrThrow()
                                    .add(branches.get(1).singletonOrThrow()));

    private ResNetV2() {}

    /** 3x3 - 3x3 basic 잔차 유닛의 사전 활성화 버전. */
    public static Block basicUnit(int inChannels, int filters, int stride) {
        SequentialBlock convPath =
                new SequentialBlock()
                        .add(Layers.conv3x3(filters, stride))
                        .add(Layers.bn())
                        .add(Activation.reluBlock())
                        .add(Layers.conv3x3(filters, 1));
        return assemble(inChannels, filters, stride, convPath);
    }

    /** 1x1 - 3x3 - 1x1 병목 잔차 유닛의 사전 활성화 버전. */
    public static Block bottleneckUnit(int inChannels, int filters, int stride) {
        int inner = filters / 4;
        SequentialBlock convPath =
                new SequentialBlock()
                        .add(Layers.conv1x1(inner, stride))
                        .add(Layers.bn())
                        .add(Activation.reluBlock())
                        .add(Layers.conv3x3(inner, 1))
                        .add(Layers.bn())
                        .add(Activation.reluBlock())
                        .add(Layers.conv1x1(filters, 1));
        return assemble(inChannels, filters, stride, convPath);
    }

    /**
     * 사전 활성화(BN - ReLU)를 어디에 붙일지 결정해 유닛을 완성한다. 지름길이 항등이냐
     * 투영이냐에 따라 블록 구조 자체가 달라진다.
     */
    private static Block assemble(
            int inChannels, int filters, int stride, SequentialBlock convPath) {
        if (inChannels == filters && stride == 1) {
            // 항등 지름길: 사전 활성화는 잔차 경로 안에만 둔다. 지름길은 x 원본이다.
            Block residual =
                    new SequentialBlock().add(Layers.bn()).add(Activation.reluBlock()).add(convPath);
            return new ParallelBlock(ADD, Arrays.asList(residual, Blocks.identityBlock()));
        }
        // 투영 지름길: 사전 활성화를 밖으로 빼서 두 경로가 함께 쓴다.
        // 투영에는 BN을 붙이지 않는다 — 항등 경로를 깨끗하게 두자는 게 v2의 취지다.
        return new SequentialBlock()
                .add(Layers.bn())
                .add(Activation.reluBlock())
                .add(
                        new ParallelBlock(
                                ADD,
                                Arrays.asList(convPath, Layers.conv1x1(filters, stride))));
    }

    /** 스펙대로 전체 네트워크를 조립한다. */
    public static SequentialBlock create(ResNetSpec spec, long numClasses) {
        SequentialBlock net = new SequentialBlock();

        if (spec.imageNetStem()) {
            // 원 구현은 ImageNet 스템에 한해 v1과 똑같이 BN - ReLU - maxpool을 둔다.
            // 첫 유닛이 다시 BN - ReLU로 시작하니 중복이지만, 논문 코드를 따른다.
            net.add(Layers.conv7x7(spec.stemFilters(), 2))
                    .add(Layers.bn())
                    .add(Activation.reluBlock())
                    .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1)));
        } else {
            // CIFAR 스템은 conv 하나뿐이다. 활성화는 첫 유닛의 사전 활성화가 맡는다.
            net.add(Layers.conv3x3(spec.stemFilters(), 1));
        }

        int inChannels = spec.stemFilters();
        for (int stage = 0; stage < spec.units().length; stage++) {
            int filters = spec.stageFilters()[stage];
            for (int unit = 0; unit < spec.units()[stage]; unit++) {
                int stride = unit == 0 ? spec.strideOf(stage) : 1;
                net.add(
                        spec.bottleneck()
                                ? bottleneckUnit(inChannels, filters, stride)
                                : basicUnit(inChannels, filters, stride));
                inChannels = filters;
            }
        }

        // 마지막 유닛이 conv로 끝나므로 풀링 전에 BN - ReLU를 한 번 더 태운다.
        // v1에는 없는 단계다.
        return net.add(Layers.bn())
                .add(Activation.reluBlock())
                .add(Pool.globalAvgPool2dBlock())
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(numClasses).build());
    }
}
