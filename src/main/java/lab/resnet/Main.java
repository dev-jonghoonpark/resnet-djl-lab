package lab.resnet;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.training.ParameterStore;

/** v1과 v2를 나란히 세워 구조와 파라미터 수를 비교한다. */
public final class Main {

    private Main() {}

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            section("잔차 유닛 하나의 구조 (basic, 64 -> 64, stride 1)");
            describeUnit(manager, "v1", ResNetV1.basicUnit(64, 64, 1), new Shape(1, 64, 56, 56));
            describeUnit(manager, "v2", ResNetV2.basicUnit(64, 64, 1), new Shape(1, 64, 56, 56));

            section("잔차 유닛 하나의 구조 (basic, 64 -> 128, stride 2 — 투영 지름길)");
            describeUnit(manager, "v1", ResNetV1.basicUnit(64, 128, 2), new Shape(1, 64, 56, 56));
            describeUnit(manager, "v2", ResNetV2.basicUnit(64, 128, 2), new Shape(1, 64, 56, 56));

            section("전체 네트워크 비교");
            System.out.printf(
                    "%-26s %14s %14s %14s%n", "모델", "v1 파라미터", "v2 파라미터", "차이");
            compare(manager, "ImageNet-18", ResNetSpec.imageNet(18), imageNet(), 1000);
            compare(manager, "ImageNet-34", ResNetSpec.imageNet(34), imageNet(), 1000);
            compare(manager, "ImageNet-50", ResNetSpec.imageNet(50), imageNet(), 1000);
            compare(manager, "CIFAR-20", ResNetSpec.cifar(20), cifar(), 10);
            compare(manager, "CIFAR-56", ResNetSpec.cifar(56), cifar(), 10);
            compare(manager, "CIFAR-164", ResNetSpec.cifar(164), cifar(), 10);

            section("실제 순전파 (CIFAR-20, 배치 4)");
            forward(manager, "v1", ResNetV1.create(ResNetSpec.cifar(20), 10), cifar(4));
            forward(manager, "v2", ResNetV2.create(ResNetSpec.cifar(20), 10), cifar(4));
        }
    }

    private static Shape imageNet() {
        return new Shape(1, 3, 224, 224);
    }

    private static Shape cifar() {
        return cifar(1);
    }

    private static Shape cifar(int batch) {
        return new Shape(batch, 3, 32, 32);
    }

    private static void compare(
            NDManager manager, String name, ResNetSpec spec, Shape input, long numClasses) {
        Block v1 = ResNetV1.create(spec, numClasses);
        Block v2 = ResNetV2.create(spec, numClasses);
        long p1 = trainableParams(manager, v1, input);
        long p2 = trainableParams(manager, v2, input);
        System.out.printf(
                "%-26s %14s %14s %+14d%n",
                name, format(p1), format(p2), p2 - p1);
    }

    /** BatchNorm의 running mean/var는 학습 대상이 아니므로 제외하고 센다. */
    private static long trainableParams(NDManager manager, Block block, Shape input) {
        block.initialize(manager, DataType.FLOAT32, input);
        return block.getParameters().values().stream()
                .filter(p -> p.requiresGradient())
                .mapToLong(p -> p.getArray().size())
                .sum();
    }

    private static void describeUnit(
            NDManager manager, String label, Block unit, Shape input) {
        unit.initialize(manager, DataType.FLOAT32, input);
        Shape output = unit.getOutputShapes(new Shape[] {input})[0];
        System.out.println("[" + label + "] " + input + " -> " + output);
        System.out.println(Blocks.describe(unit, null, 1));
        System.out.println();
    }

    private static void forward(NDManager manager, String label, Block net, Shape input) {
        net.initialize(manager, DataType.FLOAT32, input);
        try (NDManager scope = manager.newSubManager()) {
            NDArray x = scope.randomNormal(input);
            NDArray y =
                    net.forward(new ParameterStore(scope, false), new NDList(x), false)
                            .singletonOrThrow();
            System.out.printf(
                    "[%s] %s -> %s   (첫 샘플 로짓 합 %.4f)%n",
                    label, x.getShape(), y.getShape(), y.get(0).sum().getFloat());
        }
    }

    private static String format(long n) {
        return String.format("%,d", n);
    }

    private static void section(String title) {
        System.out.println();
        System.out.println("=".repeat(74));
        System.out.println("  " + title);
        System.out.println("=".repeat(74));
    }
}
