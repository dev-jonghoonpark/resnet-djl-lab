package lab.resnet;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Transform;

import java.util.Random;

/**
 * ResNet 논문이 CIFAR-10에 쓴 표준 증강: 각 변에 4픽셀 zero padding 후 32x32 랜덤 크롭,
 * 그리고 좌우 반전. 이 증강이 없으면 ResNet-20의 테스트 정확도가 8%p 이상 떨어진다.
 *
 * <p>주의할 점 하나. DJL의 {@code ArrayDataset}은 {@code BulkDataIterable}을 쓰기 때문에
 * 파이프라인이 <b>샘플 하나가 아니라 배치 전체</b>에 적용된다. 그래서 이 클래스가 받는
 * 입력은 {@code (N, 32, 32, 3)} float32 HWC 배치다.
 *
 * <p>그 결과 크롭 위치는 배치 안에서 공유된다 (오프셋 뽑기를 벡터화하기 어렵다). 매 에폭
 * 셔플되므로 한 이미지는 여러 에폭에 걸쳐 다양한 크롭을 겪게 되어 실용적으로는 충분하다.
 * 반면 좌우 반전은 마스크와 {@code where}로 <b>샘플마다 따로</b> 적용한다.
 */
final class CifarAugmentation implements Transform {

    private static final int PAD = 4;

    private final Random random;

    CifarAugmentation(long seed) {
        this.random = new Random(seed);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transform(NDArray batch) {
        return randomFlip(randomCrop(batch));
    }

    private NDArray randomCrop(NDArray batch) {
        int size = (int) batch.getShape().get(1);
        // pad는 마지막 축부터 (C, W, H) 순으로 받는다. 채널은 건드리지 않는다.
        NDArray padded = batch.pad(new Shape(0, 0, PAD, PAD, PAD, PAD), 0);
        int y = random.nextInt(2 * PAD + 1);
        int x = random.nextInt(2 * PAD + 1);
        return padded.get(String.format(":, %d:%d, %d:%d, :", y, y + size, x, x + size));
    }

    private NDArray randomFlip(NDArray batch) {
        int n = (int) batch.getShape().get(0);
        boolean[] coin = new boolean[n];
        for (int i = 0; i < n; i++) {
            coin[i] = random.nextBoolean();
        }
        NDManager manager = batch.getManager();
        NDArray mask = manager.create(coin).reshape(n, 1, 1, 1).broadcast(batch.getShape());
        NDArray flipped = batch.flip(2); // HWC이므로 너비 축은 2번이다
        return NDArrays.where(mask, flipped, batch);
    }
}
