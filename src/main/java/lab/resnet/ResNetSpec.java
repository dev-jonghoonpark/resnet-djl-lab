package lab.resnet;

/**
 * 깊이에서 스테이지 구성을 뽑아내는 표. v1과 v2가 완전히 동일하게 쓴다 —
 * 두 논문의 차이는 스테이지 구성이 아니라 잔차 유닛 내부에만 있기 때문이다.
 *
 * @param bottleneck 1x1 - 3x3 - 1x1 병목 유닛을 쓸지 (false면 3x3 - 3x3 basic 유닛)
 * @param units 스테이지별 잔차 유닛 개수
 * @param stageFilters 스테이지별 출력 채널 수
 * @param stemFilters 스템(첫 conv)의 출력 채널 수
 * @param imageNetStem 224x224용 7x7/stride2 + maxpool 스템을 쓸지 (false면 CIFAR용 3x3 스템)
 */
public record ResNetSpec(
        boolean bottleneck,
        int[] units,
        int[] stageFilters,
        int stemFilters,
        boolean imageNetStem) {

    /** ImageNet(224x224)용 4-스테이지 구성. 18 / 34 / 50 / 101 / 152를 지원한다. */
    public static ResNetSpec imageNet(int numLayers) {
        int[] basicFilters = {64, 128, 256, 512};
        int[] bottleneckFilters = {256, 512, 1024, 2048};
        return switch (numLayers) {
            case 18 -> new ResNetSpec(false, new int[] {2, 2, 2, 2}, basicFilters, 64, true);
            case 34 -> new ResNetSpec(false, new int[] {3, 4, 6, 3}, basicFilters, 64, true);
            case 50 -> new ResNetSpec(true, new int[] {3, 4, 6, 3}, bottleneckFilters, 64, true);
            case 101 -> new ResNetSpec(true, new int[] {3, 4, 23, 3}, bottleneckFilters, 64, true);
            case 152 -> new ResNetSpec(true, new int[] {3, 8, 36, 3}, bottleneckFilters, 64, true);
            default ->
                    throw new IllegalArgumentException(
                            "ImageNet ResNet은 18/34/50/101/152만 정의되어 있다: " + numLayers);
        };
    }

    /**
     * CIFAR(32x32)용 3-스테이지 구성. 논문의 6n+2(basic) / 9n+2(bottleneck) 규칙을 따른다.
     * 20, 32, 44, 56, 110은 basic이고 164, 1001은 bottleneck이다.
     */
    public static ResNetSpec cifar(int numLayers) {
        if ((numLayers - 2) % 9 == 0 && numLayers >= 164) {
            int n = (numLayers - 2) / 9;
            return new ResNetSpec(
                    true, new int[] {n, n, n}, new int[] {64, 128, 256}, 16, false);
        }
        if ((numLayers - 2) % 6 == 0) {
            int n = (numLayers - 2) / 6;
            return new ResNetSpec(
                    false, new int[] {n, n, n}, new int[] {16, 32, 64}, 16, false);
        }
        throw new IllegalArgumentException(
                "CIFAR ResNet의 깊이는 6n+2 또는 9n+2여야 한다: " + numLayers);
    }

    /** 스테이지 i의 첫 유닛이 쓰는 stride. 첫 스테이지만 해상도를 유지한다. */
    int strideOf(int stage) {
        return stage == 0 ? 1 : 2;
    }
}
