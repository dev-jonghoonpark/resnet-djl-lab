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
 * ResNet v1 — He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016).
 *
 * <p>핵심은 <b>post-activation</b>이다. 잔차 경로는 Conv - BN - ReLU 순으로 흐르고,
 * 지름길과 더한 <em>다음에</em> ReLU를 한 번 더 통과한다. 즉 항등 경로 위에도 ReLU가
 * 얹혀 있어서, 입력이 출력까지 아무 변형 없이 도달하지는 못한다. 이 점이 v2에서
 * 바뀌는 지점이다.
 *
 * <pre>
 *   x ──┬─ Conv - BN - ReLU - Conv - BN ─┬─ (+) ─ ReLU ─→
 *       └────── identity 또는 1x1 Conv ──┘
 * </pre>
 */
public final class ResNetV1 {

    /** 더한 뒤 ReLU. v1을 v1이게 만드는 한 줄이다. */
    private static final Function<List<NDList>, NDList> ADD_THEN_RELU =
            branches ->
                    new NDList(
                            Activation.relu(
                                    branches.get(0)
                                            .singletonOrThrow()
                                            .add(branches.get(1).singletonOrThrow())));

    private ResNetV1() {}

    /**
     * 3x3 - 3x3 basic 잔차 유닛. ResNet-18/34와 CIFAR 계열이 쓴다.
     *
     * @param inChannels 입력 채널 수 — 지름길에 1x1 투영이 필요한지 판단하는 데 쓴다
     * @param filters 출력 채널 수
     * @param stride 첫 conv의 stride (2면 해상도가 절반이 된다)
     */
    public static Block basicUnit(int inChannels, int filters, int stride) {
        SequentialBlock residual =
                new SequentialBlock()
                        .add(Layers.conv3x3(filters, stride))
                        .add(Layers.bn())
                        .add(Activation.reluBlock())
                        .add(Layers.conv3x3(filters, 1))
                        .add(Layers.bn());
        return new ParallelBlock(
                ADD_THEN_RELU, Arrays.asList(residual, shortcut(inChannels, filters, stride)));
    }

    /**
     * 1x1 - 3x3 - 1x1 병목 잔차 유닛. ResNet-50 이상이 쓴다. 안쪽 채널은 출력의 1/4이다.
     *
     * <p>stride는 첫 1x1에 걸린다 — 원논문 구현 그대로다. torchvision의 이른바 "v1.5"는
     * 이 stride를 가운데 3x3으로 옮겨 정확도를 조금 올렸는데, 여기서는 논문에 맞춘다.
     */
    public static Block bottleneckUnit(int inChannels, int filters, int stride) {
        int inner = filters / 4;
        SequentialBlock residual =
                new SequentialBlock()
                        .add(Layers.conv1x1(inner, stride))
                        .add(Layers.bn())
                        .add(Activation.reluBlock())
                        .add(Layers.conv3x3(inner, 1))
                        .add(Layers.bn())
                        .add(Activation.reluBlock())
                        .add(Layers.conv1x1(filters, 1))
                        .add(Layers.bn());
        return new ParallelBlock(
                ADD_THEN_RELU, Arrays.asList(residual, shortcut(inChannels, filters, stride)));
    }

    /**
     * 채널 수와 해상도가 그대로면 항등 지름길, 아니면 1x1 conv + BN 투영 지름길
     * (논문의 option B).
     */
    private static Block shortcut(int inChannels, int filters, int stride) {
        if (inChannels == filters && stride == 1) {
            return Blocks.identityBlock();
        }
        return new SequentialBlock().add(Layers.conv1x1(filters, stride)).add(Layers.bn());
    }

    /** 스펙대로 전체 네트워크를 조립한다. */
    public static SequentialBlock create(ResNetSpec spec, long numClasses) {
        SequentialBlock net = new SequentialBlock();

        // 스템. v1은 conv 직후에 BN - ReLU가 붙는다.
        if (spec.imageNetStem()) {
            net.add(Layers.conv7x7(spec.stemFilters(), 2))
                    .add(Layers.bn())
                    .add(Activation.reluBlock())
                    .add(Pool.maxPool2dBlock(new Shape(3, 3), new Shape(2, 2), new Shape(1, 1)));
        } else {
            net.add(Layers.conv3x3(spec.stemFilters(), 1)).add(Layers.bn()).add(Activation.reluBlock());
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

        return net.add(Pool.globalAvgPool2dBlock())
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder().setUnits(numClasses).build());
    }
}
