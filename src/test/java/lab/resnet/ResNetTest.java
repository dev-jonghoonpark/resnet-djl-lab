package lab.resnet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.ParameterStore;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class ResNetTest {

    private static NDManager manager;

    @BeforeAll
    static void setUp() {
        manager = NDManager.newBaseManager();
    }

    @AfterAll
    static void tearDown() {
        manager.close();
    }

    @ParameterizedTest
    @ValueSource(ints = {18, 34, 50, 101})
    void imageNetShapes(int numLayers) {
        ResNetSpec spec = ResNetSpec.imageNet(numLayers);
        Shape input = new Shape(2, 3, 224, 224);
        assertEquals(new Shape(2, 1000), outputShape(ResNetV1.create(spec, 1000), input));
        assertEquals(new Shape(2, 1000), outputShape(ResNetV2.create(spec, 1000), input));
    }

    @ParameterizedTest
    @ValueSource(ints = {20, 32, 56, 110, 164})
    void cifarShapes(int numLayers) {
        ResNetSpec spec = ResNetSpec.cifar(numLayers);
        Shape input = new Shape(2, 3, 32, 32);
        assertEquals(new Shape(2, 10), outputShape(ResNetV1.create(spec, 10), input));
        assertEquals(new Shape(2, 10), outputShape(ResNetV2.create(spec, 10), input));
    }

    @Test
    void downsamplingUnitHalvesResolution() {
        Shape input = new Shape(2, 64, 56, 56);
        Shape expected = new Shape(2, 128, 28, 28);
        assertEquals(expected, outputShape(ResNetV1.basicUnit(64, 128, 2), input));
        assertEquals(expected, outputShape(ResNetV2.basicUnit(64, 128, 2), input));
        assertEquals(expected, outputShape(ResNetV1.bottleneckUnit(64, 128, 2), input));
        assertEquals(expected, outputShape(ResNetV2.bottleneckUnit(64, 128, 2), input));
    }

    /**
     * v1은 덧셈 뒤에 ReLU가 있으므로 유닛의 출력이 절대 음수가 될 수 없다. 이 성질 하나가
     * post-activation의 관측 가능한 증거다.
     */
    @Test
    void v1UnitOutputIsNeverNegative() {
        assertTrue(minOutput(ResNetV1.basicUnit(64, 64, 1)) >= 0f);
        assertTrue(minOutput(ResNetV1.bottleneckUnit(64, 64, 1)) >= 0f);
    }

    /**
     * v2는 덧셈이 마지막이므로 음수가 그대로 통과한다. 항등 경로가 아무 변형 없이
     * 살아있다는 뜻이다.
     */
    @Test
    void v2UnitOutputPassesNegativesThrough() {
        assertTrue(minOutput(ResNetV2.basicUnit(64, 64, 1)) < 0f);
        assertTrue(minOutput(ResNetV2.bottleneckUnit(64, 64, 1)) < 0f);
    }

    /**
     * 채널이 그대로인 v2 유닛에서 잔차 경로의 마지막 conv 가중치를 0으로 만들면 출력이
     * 입력과 정확히 같아야 한다. 항등 사상이 실제로 성립하는지에 대한 직접 검증이다.
     */
    @Test
    void v2IdentityUnitIsExactIdentityWhenResidualIsZeroed() {
        Shape input = new Shape(2, 16, 8, 8);
        Block unit = ResNetV2.basicUnit(16, 16, 1);
        unit.initialize(manager, DataType.FLOAT32, input);

        // 잔차 경로의 conv 가중치를 전부 0으로 — 잔차 출력이 0이 된다.
        unit.getParameters().values().stream()
                .filter(p -> p.getName().equals("weight") && p.getArray().getShape().dimension() == 4)
                .forEach(p -> p.getArray().muli(0f));

        try (NDManager scope = manager.newSubManager()) {
            NDArray x = scope.randomNormal(input);
            NDArray y =
                    unit.forward(new ParameterStore(scope, false), new NDList(x), false)
                            .singletonOrThrow();
            assertEquals(0f, y.sub(x).abs().max().getFloat(), 1e-6f);
        }
    }

    @Test
    void augmentationPreservesBatchShape() {
        NDArray batch = manager.randomUniform(0f, 255f, new Shape(8, 32, 32, 3));
        assertEquals(batch.getShape(), new CifarAugmentation(1).transform(batch).getShape());
    }

    /**
     * 크롭 오프셋은 배치가 공유하지만 좌우 반전은 샘플마다 따로 걸린다. 같은 이미지를
     * 복제한 배치를 넣으면 출력이 한 종류로 뭉치지 않아야 한다.
     */
    @Test
    void augmentationFlipsPerSample() {
        NDArray one = manager.randomUniform(0f, 255f, new Shape(1, 32, 32, 3));
        NDArray batch = one.repeat(0, 32);
        NDArray out = new CifarAugmentation(7).transform(batch);

        boolean anyDifferent = false;
        for (int i = 1; i < 32 && !anyDifferent; i++) {
            anyDifferent = out.get(i).sub(out.get(0)).abs().max().getFloat() > 1e-6f;
        }
        assertTrue(anyDifferent, "반전이 샘플마다 적용되어야 한다");
    }

    @Test
    void invalidDepthsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> ResNetSpec.imageNet(42));
        assertThrows(IllegalArgumentException.class, () -> ResNetSpec.cifar(21));
    }

    private static Shape outputShape(Block block, Shape input) {
        block.initialize(manager, DataType.FLOAT32, input);
        return block.getOutputShapes(new Shape[] {input})[0];
    }

    private static float minOutput(Block unit) {
        Shape input = new Shape(2, 64, 8, 8);
        unit.initialize(manager, DataType.FLOAT32, input);
        try (NDManager scope = manager.newSubManager()) {
            NDArray x = scope.randomNormal(input);
            return unit.forward(new ParameterStore(scope, false), new NDList(x), false)
                    .singletonOrThrow()
                    .min()
                    .getFloat();
        }
    }
}
