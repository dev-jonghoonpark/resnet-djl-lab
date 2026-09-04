package lab.resnet;

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.norm.BatchNorm;

/**
 * v1과 v2가 공유하는 부품. 두 버전의 차이가 "부품"이 아니라 "조립 순서"에 있다는 걸
 * 드러내려고 일부러 여기에 모아두었다.
 */
final class Layers {

    /** 원 논문 구현(He et al.의 MXNet 코드)이 쓰는 값. */
    static final float BN_EPS = 2e-5f;

    static final float BN_MOMENTUM = 0.9f;

    private Layers() {}

    /** Conv 뒤에는 항상 BatchNorm이 오므로 bias는 필요 없다 (BN의 beta가 그 역할을 한다). */
    static Block conv(int filters, int kernel, int stride, int padding) {
        return Conv2d.builder()
                .setKernelShape(new Shape(kernel, kernel))
                .setFilters(filters)
                .optStride(new Shape(stride, stride))
                .optPadding(new Shape(padding, padding))
                .optBias(false)
                .build();
    }

    static Block conv1x1(int filters, int stride) {
        return conv(filters, 1, stride, 0);
    }

    static Block conv3x3(int filters, int stride) {
        return conv(filters, 3, stride, 1);
    }

    static Block conv7x7(int filters, int stride) {
        return conv(filters, 7, stride, 3);
    }

    static Block bn() {
        return BatchNorm.builder().optEpsilon(BN_EPS).optMomentum(BN_MOMENTUM).build();
    }
}
