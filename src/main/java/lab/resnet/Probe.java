package lab.resnet;

import ai.djl.basicdataset.cv.classification.Cifar10;
import ai.djl.engine.Engine;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.BlockList;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.Pair;

import java.io.IOException;
import java.math.BigDecimal;
import java.math.MathContext;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 이미지 <b>한 장</b>이 학습 한 사이클(순전파 → 손실 → 역전파 → 갱신 → 재순전파)을
 * 도는 동안 실제로 어떤 값이 오가는지 전부 기록해 JSON으로 뱉는 계측기.
 *
 * <pre>
 *   mvn exec:java -Dexec.mainClass=lab.resnet.Probe -Dexec.args="--version v1"
 * </pre>
 *
 * <p>{@link Train}과 달리 학습이 목적이 아니다. 그래서 배치도 1이다 —
 * <b>BatchNorm의 평균·분산이 그 한 장에서만 나온다</b>는 뜻이고, 실제 학습(배치 128)과
 * 다른 유일한 지점이다. 나머지(초기화, 손실, 옵티마이저 설정)는 {@code Train}과 같다.
 *
 * <p>기록 방식은 블록 트리를 직접 걸어 내려가면서 자식마다 forward를 따로 호출하는 것이다.
 * {@code SequentialBlock.forward}에 맡기면 중간값을 볼 수 없기 때문이다. {@link ParallelBlock}의
 * 합류 함수는 private이라 꺼낼 수 없어서 버전별로 알고 있는 규칙(v1은 더한 뒤 ReLU,
 * v2는 그냥 더하기)을 여기서 다시 적용한다.
 */
public final class Probe {

    private static final String[] CLASSES = {
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    };

    /** 히트맵으로 뽑아낼 채널 수. 전부 담으면 JSON이 수십 MB가 된다. */
    private static final int MAX_MAPS = 4;

    private final boolean v1;
    private final int label;
    private final ParameterStore ps;
    private final List<Map<String, Object>> steps = new ArrayList<>();
    private final Map<String, Object> exhibits = new LinkedHashMap<>();

    private String group = "stem";
    private int stepId;

    private Probe(boolean v1, int label, ParameterStore ps) {
        this.v1 = v1;
        this.label = label;
        this.ps = ps;
    }

    public static void main(String[] args) throws IOException {
        Opt opt = Opt.parse(args);
        Files.createDirectories(Paths.get(opt.outDir));
        for (String version : opt.versions) {
            Path out = Paths.get(opt.outDir).resolve("trace-" + version + ".json");
            String json = run(version, opt);
            Files.writeString(out, json, StandardCharsets.UTF_8);
            System.out.printf("%n기록 완료: %s (%,d bytes)%n%n", out, json.length());
        }
    }

    private static String run(String version, Opt opt) throws IOException {
        boolean v1 = "v1".equals(version);
        Engine engine = Engine.getInstance();
        engine.setRandomSeed((int) opt.seed);

        ResNetSpec spec = ResNetSpec.cifar(opt.depth);
        SequentialBlock net = v1 ? ResNetV1.create(spec, 10) : ResNetV2.create(spec, 10);

        Map<String, Object> root = new LinkedHashMap<>();

        try (NDManager manager = engine.newBaseManager()) {
            Shape inputShape = new Shape(1, 3, 32, 32);
            net.setInitializer(
                    new XavierInitializer(
                            XavierInitializer.RandomType.GAUSSIAN,
                            XavierInitializer.FactorType.IN,
                            2f),
                    Parameter.Type.WEIGHT);
            net.initialize(manager, DataType.FLOAT32, inputShape);

            // ── 0단계: 픽셀 읽기 → ToTensor → Normalize ────────────────────────────
            Path imagePath = Paths.get(opt.image);
            int label = labelOf(imagePath);
            Image image = ImageFactory.getInstance().fromFile(imagePath);
            NDArray raw = image.toNDArray(manager); // (H, W, C), uint8
            NDArray rawF = raw.toType(DataType.FLOAT32, false); // 통계를 내려면 float이어야 한다
            NDArray tensor = new ToTensor().transform(raw); // (C, H, W), float 0~1
            NDArray norm =
                    new Normalize(Cifar10.NORMALIZE_MEAN, Cifar10.NORMALIZE_STD).transform(tensor);
            NDArray x = norm.expandDims(0); // (1, C, H, W)

            root.put("version", version);
            root.put("depth", opt.depth);
            root.put("spec", specInfo(spec));
            root.put("seed", opt.seed);
            root.put("image", imageInfo(imagePath, label, rawF));
            root.put("preprocess", preprocess(rawF, tensor, norm));

            Probe probe = new Probe(v1, label, new ParameterStore(manager, false));

            Loss loss = Loss.softmaxCrossEntropyLoss();
            NDArray labelArray = manager.create(new float[] {label}, new Shape(1));

            NDArray logits;
            NDArray lossValue;
            try (GradientCollector gc = engine.newGradientCollector()) {
                for (Pair<String, Parameter> p : net.getParameters()) {
                    if (p.getValue().requiresGradient()) {
                        p.getValue().getArray().setRequiresGradient(true);
                    }
                }

                // ── 1단계: 순전파. 블록 하나하나 값을 남기며 내려간다 ──────────────
                logits = probe.forwardNetwork(net, spec, x).singletonOrThrow();

                // ── 2단계: 소프트맥스와 손실 ─────────────────────────────────────
                lossValue = loss.evaluate(new NDList(labelArray), new NDList(logits));
                root.put("head", head(logits, label, lossValue.getFloat()));

                // ── 3단계: 역전파 ────────────────────────────────────────────────
                gc.backward(lossValue);
            }

            root.put("steps", probe.steps);
            root.put("exhibits", probe.exhibits);
            root.put("backward", backward(net, logits, label, probe));

            // ── 4단계: 가중치 갱신 ───────────────────────────────────────────────
            root.put("update", update(net, opt));

            // ── 5단계: 같은 이미지로 다시 순전파 ─────────────────────────────────
            NDArray logits2 =
                    net.forward(new ParameterStore(manager, false), new NDList(x), true)
                            .singletonOrThrow();
            float loss2 =
                    loss.evaluate(new NDList(labelArray), new NDList(logits2)).getFloat();
            Map<String, Object> after = head(logits2, label, loss2);
            after.put("lossBefore", lossValue.getFloat());
            after.put("deltaLoss", loss2 - lossValue.getFloat());
            root.put("after", after);

            console(version, root, label, logits, lossValue.getFloat(), logits2, loss2);
        }
        return Json.write(root);
    }

    // ────────────────────────────────────────────────────────────────────────────
    // 순전파 추적
    // ────────────────────────────────────────────────────────────────────────────

    /**
     * 최상위 SequentialBlock을 직접 훑으면서 스템 / 잔차 유닛 / 헤드로 구간을 나눈다.
     * 구간 경계는 버전마다 다르다 — v1 스템은 conv-bn-relu 세 블록, v2 스템은 conv 하나다.
     */
    private NDList forwardNetwork(SequentialBlock net, ResNetSpec spec, NDArray input) {
        BlockList children = net.getChildren();
        int stemLen = v1 ? 3 : 1;
        int totalUnits = Arrays.stream(spec.units()).sum();

        int[] stageOf = new int[totalUnits];
        int[] unitOf = new int[totalUnits];
        for (int i = 0, u = 0; i < spec.units().length; i++) {
            for (int j = 0; j < spec.units()[i]; j++, u++) {
                stageOf[u] = i + 1;
                unitOf[u] = j;
            }
        }

        NDList cur = new NDList(input);
        for (int i = 0; i < children.size(); i++) {
            if (i < stemLen) {
                group = "stem";
            } else if (i < stemLen + totalUnits) {
                int u = i - stemLen;
                group = "stage" + stageOf[u] + "/unit" + unitOf[u];
            } else {
                group = "head";
            }
            cur = walk(children.get(i).getValue(), children.get(i).getKey(), cur, 0);
        }
        return cur;
    }

    private NDList walk(Block block, String path, NDList input, int depth) {
        BlockList children = block.getChildren();

        if (block instanceof ParallelBlock) {
            List<NDList> branches = new ArrayList<>();
            for (int i = 0; i < children.size(); i++) {
                String name = i == 0 ? "잔차 경로" : "지름길";
                branches.add(
                        walk(
                                children.get(i).getValue(),
                                path + "/" + name + ":" + children.get(i).getKey(),
                                input,
                                depth + 1));
            }
            NDArray a = branches.get(0).singletonOrThrow();
            NDArray b = branches.get(1).singletonOrThrow();
            NDArray sum = a.add(b);
            record(path + "/(+)", "덧셈", "Add", depth, input.singletonOrThrow(), sum, null, 0)
                    .put("note", "잔차 경로 + 지름길");
            residualExhibit(a, b, sum);
            if (v1) {
                NDArray relu = Activation.relu(sum);
                record(path + "/ReLU", "ReLU", "ReLU", depth, sum, relu, null, 0)
                        .put("note", "v1은 더한 뒤에 ReLU를 한 번 더 통과한다");
                return new NDList(relu);
            }
            return new NDList(sum);
        }

        if (!children.isEmpty()) { // SequentialBlock
            NDList cur = input;
            for (Pair<String, Block> child : children) {
                cur = walk(child.getValue(), path + "/" + child.getKey(), cur, depth + 1);
            }
            return cur;
        }

        // 잎 블록 — 여기서만 실제 연산이 일어난다.
        NDArray in = input.singletonOrThrow();
        NDList out = block.forward(ps, input, true);
        NDArray o = out.singletonOrThrow();
        String kind = kindOf(block);
        Map<String, Object> step =
                record(
                        path,
                        labelOf(block, kind, o),
                        kind,
                        depth,
                        in,
                        o,
                        block,
                        macs(block, kind, in, o));

        switch (kind) {
            case "Conv2d" -> convExhibit(block, in, o, path);
            case "BatchNorm" -> bnExhibit(block, in, o, path);
            case "ReLU" -> reluExhibit(in, o, path);
            case "globalAvgPool2d" -> gapExhibit(in, o);
            case "Linear" -> {
                step.put("note", "64차원 특징 → 클래스 10개 점수");
                linearExhibit(block, in, o, path);
            }
            case "identity" -> step.put("note", "입력을 그대로 통과시킨다 (항등 지름길)");
            default -> {}
        }
        return out;
    }

    private Map<String, Object> record(
            String path,
            String label,
            String kind,
            int depth,
            NDArray in,
            NDArray out,
            Block block,
            long macs) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("i", stepId++);
        m.put("group", group);
        m.put("depth", depth);
        m.put("path", path);
        m.put("label", label);
        m.put("kind", kind);
        m.put("in", out(in.getShape()));
        m.put("out", out(out.getShape()));
        m.put("stats", stats(out));
        if (macs > 0) {
            m.put("macs", macs);
        }
        if (block != null && !block.getParameters().isEmpty()) {
            List<Map<String, Object>> params = new ArrayList<>();
            for (Pair<String, Parameter> p : block.getParameters()) {
                Map<String, Object> pm = new LinkedHashMap<>();
                pm.put("name", p.getKey());
                pm.put("shape", out(p.getValue().getArray().getShape()));
                pm.put("count", p.getValue().getArray().size());
                pm.put("trainable", p.getValue().requiresGradient());
                params.add(pm);
            }
            m.put("params", params);
        }
        if (out.getShape().dimension() == 4) {
            m.put("maps", maps(out));
        } else if (out.size() <= 128) {
            m.put("values", floats(out.toFloatArray()));
        }
        steps.add(m);
        return m;
    }

    // ────────────────────────────────────────────────────────────────────────────
    // 실제 산수를 손으로 다시 해보는 전시물들
    // ────────────────────────────────────────────────────────────────────────────

    /** 첫 conv에서 출력 한 칸이 어떻게 계산되는지 곱셈 27개를 전부 펼친다. */
    private void convExhibit(Block block, NDArray in, NDArray out, String path) {
        if (exhibits.containsKey("conv")) {
            return;
        }
        NDArray w = block.getParameters().get(0).getValue().getArray();
        long[] ws = w.getShape().getShape();
        int k = (int) ws[2];
        if (k != 3) {
            return;
        }
        int oc = 0;
        int oy = 16;
        int ox = 16;
        int inC = (int) ws[1];

        List<Map<String, Object>> terms = new ArrayList<>();
        double sum = 0;
        for (int c = 0; c < inC; c++) {
            for (int dy = 0; dy < k; dy++) {
                for (int dx = 0; dx < k; dx++) {
                    float xv = in.getFloat(0, c, oy - 1 + dy, ox - 1 + dx);
                    float wv = w.getFloat(oc, c, dy, dx);
                    sum += (double) xv * wv;
                    Map<String, Object> t = new LinkedHashMap<>();
                    t.put("c", c);
                    t.put("dy", dy);
                    t.put("dx", dx);
                    t.put("x", xv);
                    t.put("w", wv);
                    t.put("xw", xv * wv);
                    terms.add(t);
                }
            }
        }
        Map<String, Object> e = new LinkedHashMap<>();
        e.put("path", path);
        e.put("outChannel", oc);
        e.put("y", oy);
        e.put("x", ox);
        e.put("kernelShape", out(w.getShape()));
        e.put("terms", terms);
        e.put("sum", sum);
        e.put("engine", out.getFloat(0, oc, oy, ox));
        e.put("bias", false);
        e.put(
                "note",
                "Conv에는 bias가 없다 — 바로 뒤 BatchNorm의 beta가 그 역할을 하기 때문이다.");
        exhibits.put("conv", e);
    }

    /** BatchNorm 한 채널의 평균·분산을 직접 구해 정규화 식을 그대로 재현한다. */
    private void bnExhibit(Block block, NDArray in, NDArray out, String path) {
        if (exhibits.containsKey("bn")) {
            return;
        }
        int ch = 0;
        int oy = 16;
        int ox = 16;
        float[] plane = in.get(new NDIndex(0, ch)).toFloatArray();
        double mean = 0;
        for (float v : plane) {
            mean += v;
        }
        mean /= plane.length;
        double var = 0;
        for (float v : plane) {
            var += (v - mean) * (v - mean);
        }
        var /= plane.length; // BatchNorm은 편향 분산(1/N)을 쓴다

        float gamma = param(block, "gamma").getFloat(ch);
        float beta = param(block, "beta").getFloat(ch);
        float xv = in.getFloat(0, ch, oy, ox);
        double xhat = (xv - mean) / Math.sqrt(var + Layers.BN_EPS);
        double y = gamma * xhat + beta;

        Map<String, Object> e = new LinkedHashMap<>();
        e.put("path", path);
        e.put("channel", ch);
        e.put("y", oy);
        e.put("x", ox);
        e.put("n", plane.length);
        e.put("mean", mean);
        e.put("var", var);
        e.put("std", Math.sqrt(var + Layers.BN_EPS));
        e.put("eps", Layers.BN_EPS);
        e.put("gamma", gamma);
        e.put("beta", beta);
        e.put("input", xv);
        e.put("xhat", xhat);
        e.put("result", y);
        e.put("engine", out.getFloat(0, ch, oy, ox));
        e.put("runningMean", param(block, "runningMean").getFloat(ch));
        e.put("runningVar", param(block, "runningVar").getFloat(ch));
        e.put("momentum", Layers.BN_MOMENTUM);
        e.put(
                "note",
                "배치가 1장이라 평균·분산이 이 이미지의 32x32=1024칸에서만 나온다."
                        + " 실제 학습은 배치 128장 전체에서 뽑는다.");
        exhibits.put("bn", e);
    }

    private void reluExhibit(NDArray in, NDArray out, String path) {
        if (exhibits.containsKey("relu")) {
            return;
        }
        float[] a = in.toFloatArray();
        int killed = 0;
        for (float v : a) {
            if (v <= 0) {
                killed++;
            }
        }
        List<Map<String, Object>> samples = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            int idx = i * (a.length / 12);
            Map<String, Object> s = new LinkedHashMap<>();
            s.put("in", a[idx]);
            s.put("out", Math.max(0f, a[idx]));
            samples.add(s);
        }
        Map<String, Object> e = new LinkedHashMap<>();
        e.put("path", path);
        e.put("total", a.length);
        e.put("killed", killed);
        e.put("killedFrac", (double) killed / a.length);
        e.put("samples", samples);
        e.put("note", "음수를 0으로 눌러 버린다. 여기서 죽은 칸은 역전파 때 그래디언트도 0이다.");
        exhibits.put("relu", e);
    }

    /** 첫 잔차 유닛의 합류 지점 — 지름길이 실제로 무엇을 더하는지 본다. */
    private void residualExhibit(NDArray residual, NDArray shortcut, NDArray sum) {
        if (exhibits.containsKey("residual")) {
            return;
        }
        int ch = 0;
        int oy = 16;
        int ox = 16;
        if (sum.getShape().dimension() != 4
                || sum.getShape().get(2) <= oy
                || sum.getShape().get(3) <= ox) {
            return;
        }
        float r = residual.getFloat(0, ch, oy, ox);
        float s = shortcut.getFloat(0, ch, oy, ox);
        Map<String, Object> e = new LinkedHashMap<>();
        e.put("channel", ch);
        e.put("y", oy);
        e.put("x", ox);
        e.put("residual", r);
        e.put("shortcut", s);
        e.put("sum", r + s);
        e.put("engine", sum.getFloat(0, ch, oy, ox));
        e.put("afterRelu", v1 ? Math.max(0f, r + s) : null);
        e.put("residualStats", stats(residual));
        e.put("shortcutStats", stats(shortcut));
        e.put("sumStats", stats(sum));
        e.put(
                "note",
                v1
                        ? "v1은 여기서 ReLU를 한 번 더 통과한다 — 지름길로 온 값도 음수면 잘린다."
                        : "v2는 여기가 끝이다. 지름길 값이 손대지 않은 채 다음 유닛으로 간다.");
        exhibits.put("residual", e);
    }

    /** 전역 평균 풀링 — 8x8 = 64칸이 숫자 하나가 되는 과정. */
    private void gapExhibit(NDArray in, NDArray out) {
        if (exhibits.containsKey("gap")) {
            return;
        }
        int ch = 0;
        float[] plane = in.get(new NDIndex(0, ch)).toFloatArray();
        double sum = 0;
        for (float v : plane) {
            sum += v;
        }
        Map<String, Object> e = new LinkedHashMap<>();
        e.put("channel", ch);
        e.put("shape", out(in.getShape()));
        e.put("values", floats(plane));
        e.put("sum", sum);
        e.put("mean", sum / plane.length);
        e.put("engine", out.toFloatArray()[0]);
        e.put("pooled", floats(out.toFloatArray()));
        e.put("note", "채널마다 공간 정보를 평균 하나로 접는다. 64채널 -> 길이 64 벡터.");
        exhibits.put("gap", e);
    }

    /** 마지막 Linear — 정답 클래스의 점수 하나가 64개 곱의 합으로 어떻게 만들어지는지. */
    private void linearExhibit(Block block, NDArray in, NDArray out, String path) {
        if (exhibits.containsKey("linear")) {
            return;
        }
        NDArray w = param(block, "weight"); // (10, 64)
        NDArray b = param(block, "bias"); // (10)
        int cols = (int) w.getShape().get(1);
        float[] x = in.toFloatArray();

        List<Map<String, Object>> terms = new ArrayList<>();
        double sum = 0;
        for (int j = 0; j < cols; j++) {
            float wv = w.getFloat(label, j);
            sum += (double) wv * x[j];
            Map<String, Object> t = new LinkedHashMap<>();
            t.put("j", j);
            t.put("x", x[j]);
            t.put("w", wv);
            t.put("xw", wv * x[j]);
            terms.add(t);
        }
        float bias = b.getFloat(label);

        Map<String, Object> e = new LinkedHashMap<>();
        e.put("path", path);
        e.put("class", label);
        e.put("className", CLASSES[label]);
        e.put("terms", terms);
        e.put("sum", sum);
        e.put("bias", bias);
        e.put("result", sum + bias);
        e.put("engine", out.getFloat(0, label));
        e.put("weightRow", floats(w.get(new NDIndex(label)).toFloatArray()));
        e.put("note", "클래스 10개 각각에 대해 이 64항 내적을 한 번씩 한다.");
        exhibits.put("linear", e);
    }

    // ────────────────────────────────────────────────────────────────────────────
    // 손실 · 역전파 · 갱신
    // ────────────────────────────────────────────────────────────────────────────

    private static Map<String, Object> head(NDArray logits, int label, float lossValue) {
        float[] z = logits.toFloatArray();
        double max = Double.NEGATIVE_INFINITY;
        for (float v : z) {
            max = Math.max(max, v);
        }
        double[] exp = new double[z.length];
        double denom = 0;
        for (int i = 0; i < z.length; i++) {
            exp[i] = Math.exp(z[i] - max);
            denom += exp[i];
        }
        double[] p = new double[z.length];
        int pred = 0;
        for (int i = 0; i < z.length; i++) {
            p[i] = exp[i] / denom;
            if (z[i] > z[pred]) {
                pred = i;
            }
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("classes", CLASSES);
        m.put("label", label);
        m.put("labelName", CLASSES[label]);
        m.put("logits", floats(z));
        m.put("shift", max);
        m.put("exp", exp);
        m.put("denom", denom);
        m.put("probs", p);
        m.put("pred", pred);
        m.put("predName", CLASSES[pred]);
        m.put("pTrue", p[label]);
        m.put("loss", -Math.log(p[label]));
        m.put("lossFromDjl", lossValue);
        return m;
    }

    /**
     * 역전파 결과를 모은다. 손실의 로짓에 대한 미분은 손으로 적을 수 있을 만큼 간단하다
     * — softmax 확률에서 정답 자리만 1을 뺀 값이다. 마지막 Linear의 가중치 그래디언트를
     * 그 식으로 직접 계산해 엔진이 준 값과 맞춰 본다.
     */
    private static Map<String, Object> backward(
            SequentialBlock net, NDArray logits, int label, Probe probe) {
        float[] z = logits.toFloatArray();
        double max = Double.NEGATIVE_INFINITY;
        for (float v : z) {
            max = Math.max(max, v);
        }
        double denom = 0;
        for (float v : z) {
            denom += Math.exp(v - max);
        }
        double[] dz = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            dz[i] = Math.exp(z[i] - max) / denom - (i == label ? 1 : 0);
        }

        Map<String, Object> m = new LinkedHashMap<>();
        m.put("dLogits", dz);
        m.put(
                "dLogitsNote",
                "dL/dz = softmax(z) - onehot(label). 정답 클래스만 확률에서 1을 뺀다."
                        + " 나머지는 확률 그대로 — 점수를 낮추라는 뜻이다.");

        // 마지막 Linear 가중치의 그래디언트를 손계산과 대조한다.
        Parameter linearWeight = null;
        for (Pair<String, Parameter> p : net.getParameters()) {
            if (p.getKey().endsWith("weight") && p.getValue().getArray().getShape().dimension() == 2) {
                linearWeight = p.getValue();
            }
        }
        Object pooledObj = probe.exhibits.containsKey("gap")
                ? ((Map<?, ?>) probe.exhibits.get("gap")).get("pooled")
                : null;
        if (linearWeight != null && pooledObj instanceof double[] pooled) {
            NDArray g = linearWeight.getArray().getGradient();
            float[] gf = g.toFloatArray();
            long cols = g.getShape().get(1);
            double worst = 0;
            List<Map<String, Object>> check = new ArrayList<>();
            for (int k = 0; k < dz.length; k++) {
                for (int j = 0; j < cols; j++) {
                    double manual = dz[k] * pooled[j];
                    double engine = gf[(int) (k * cols + j)];
                    worst = Math.max(worst, Math.abs(manual - engine));
                    if (j < 4 && k < 3) {
                        Map<String, Object> c = new LinkedHashMap<>();
                        c.put("k", k);
                        c.put("j", j);
                        c.put("dz", dz[k]);
                        c.put("x", pooled[j]);
                        c.put("manual", manual);
                        c.put("engine", engine);
                        check.add(c);
                    }
                }
            }
            Map<String, Object> lc = new LinkedHashMap<>();
            lc.put("formula", "dL/dW[k][j] = dL/dz[k] * pooled[j]");
            lc.put("samples", check);
            lc.put("maxAbsDiff", worst);
            m.put("linearGradCheck", lc);
        }

        m.put("layers", paramTable(net, true));
        return m;
    }

    /** {@link Train}과 똑같은 옵티마이저로 한 스텝 밟고, 무엇이 얼마나 움직였는지 남긴다. */
    private static Map<String, Object> update(SequentialBlock net, Opt opt) {
        Optimizer sgd =
                Optimizer.sgd()
                        .setLearningRateTracker(Tracker.fixed(opt.lr))
                        .optMomentum(0.9f)
                        .optWeightDecays(1e-4f)
                        .build();

        Map<String, float[]> before = new LinkedHashMap<>();
        Map<String, float[]> grads = new LinkedHashMap<>();
        for (Pair<String, Parameter> p : net.getParameters()) {
            if (!p.getValue().requiresGradient()) {
                continue;
            }
            before.put(p.getKey(), p.getValue().getArray().toFloatArray());
            grads.put(p.getKey(), p.getValue().getArray().getGradient().toFloatArray());
        }
        for (Pair<String, Parameter> p : net.getParameters()) {
            if (!p.getValue().requiresGradient()) {
                continue;
            }
            sgd.update(
                    p.getValue().getId(),
                    p.getValue().getArray(),
                    p.getValue().getArray().getGradient());
        }

        Map<String, Object> m = new LinkedHashMap<>();
        m.put("lr", opt.lr);
        m.put("momentum", 0.9f);
        m.put("weightDecay", 1e-4f);
        m.put(
                "formula",
                "첫 스텝은 momentum 상태가 0이라 w <- w - lr * (dL/dw + wd * w) 그대로다.");

        List<Map<String, Object>> rows = new ArrayList<>();
        Map<String, Object> sample = null;
        for (Pair<String, Parameter> p : net.getParameters()) {
            if (!p.getValue().requiresGradient()) {
                continue;
            }
            float[] b = before.get(p.getKey());
            float[] g = grads.get(p.getKey());
            float[] a = p.getValue().getArray().toFloatArray();
            double dAbs = 0;
            double dMax = 0;
            double wAbs = 0;
            for (int i = 0; i < a.length; i++) {
                double d = Math.abs(a[i] - b[i]);
                dAbs += d;
                dMax = Math.max(dMax, d);
                wAbs += Math.abs(b[i]);
            }
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("name", p.getKey());
            row.put("count", a.length);
            row.put("deltaAbsMean", dAbs / a.length);
            row.put("deltaMax", dMax);
            row.put("relative", wAbs == 0 ? 0 : dAbs / wAbs);
            rows.add(row);

            if (sample == null && a.length > 0) {
                sample = new LinkedHashMap<>();
                sample.put("name", p.getKey());
                sample.put("index", 0);
                sample.put("w", b[0]);
                sample.put("grad", g[0]);
                sample.put("decay", 1e-4f * b[0]);
                sample.put("lr", opt.lr);
                sample.put("predicted", b[0] - opt.lr * (g[0] + 1e-4f * b[0]));
                sample.put("actual", a[0]);
            }
        }
        m.put("exhibit", sample);
        m.put("layers", rows);
        return m;
    }

    private static List<Map<String, Object>> paramTable(SequentialBlock net, boolean withGrad) {
        List<Map<String, Object>> rows = new ArrayList<>();
        for (Pair<String, Parameter> p : net.getParameters()) {
            Parameter param = p.getValue();
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("name", p.getKey());
            row.put("shape", out(param.getArray().getShape()));
            row.put("count", param.getArray().size());
            row.put("trainable", param.requiresGradient());
            row.put("weightNorm", norm(param.getArray().toFloatArray()));
            if (withGrad && param.requiresGradient()) {
                try {
                    float[] g = param.getArray().getGradient().toFloatArray();
                    row.put("gradNorm", norm(g));
                    row.put("gradAbsMean", absMean(g));
                    row.put("gradMax", absMax(g));
                } catch (RuntimeException ignored) {
                    row.put("gradNorm", null);
                }
            }
            rows.add(row);
        }
        return rows;
    }

    // ────────────────────────────────────────────────────────────────────────────
    // 입력 전처리 기록
    // ────────────────────────────────────────────────────────────────────────────

    private static Map<String, Object> imageInfo(Path path, int label, NDArray raw) {
        long[] s = raw.getShape().getShape();
        int h = (int) s[0];
        int w = (int) s[1];
        float[] flat = raw.toFloatArray();
        byte[] rgb = new byte[h * w * 3];
        for (int i = 0; i < rgb.length; i++) {
            rgb[i] = (byte) Math.round(flat[i]);
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("path", path.toString());
        m.put("label", label);
        m.put("labelName", CLASSES[label]);
        m.put("height", h);
        m.put("width", w);
        m.put("rgb", Base64.getEncoder().encodeToString(rgb));
        return m;
    }

    private static Map<String, Object> preprocess(NDArray raw, NDArray tensor, NDArray norm) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("mean", Cifar10.NORMALIZE_MEAN);
        m.put("std", Cifar10.NORMALIZE_STD);
        m.put("rawShape", out(raw.getShape()));
        m.put("tensorShape", out(tensor.getShape()));

        int py = 16;
        int px = 16;
        List<Map<String, Object>> pixel = new ArrayList<>();
        String[] names = {"R", "G", "B"};
        for (int c = 0; c < 3; c++) {
            Map<String, Object> pm = new LinkedHashMap<>();
            pm.put("channel", names[c]);
            pm.put("raw", raw.getFloat(py, px, c));
            pm.put("scaled", tensor.getFloat(c, py, px));
            pm.put("mean", Cifar10.NORMALIZE_MEAN[c]);
            pm.put("std", Cifar10.NORMALIZE_STD[c]);
            pm.put("normalized", norm.getFloat(c, py, px));
            pixel.add(pm);
        }
        m.put("pixelY", py);
        m.put("pixelX", px);
        m.put("pixel", pixel);

        List<Map<String, Object>> planes = new ArrayList<>();
        for (int c = 0; c < 3; c++) {
            Map<String, Object> pm = plane(norm.get(new NDIndex(c)));
            pm.put("channel", names[c]);
            pm.put("stats", stats(norm.get(new NDIndex(c))));
            planes.add(pm);
        }
        m.put("planes", planes);
        m.put("statsRaw", stats(raw));
        m.put("statsTensor", stats(tensor));
        m.put("statsNorm", stats(norm));
        return m;
    }

    // ────────────────────────────────────────────────────────────────────────────
    // 잡다한 도구들
    // ────────────────────────────────────────────────────────────────────────────

    private static NDArray param(Block block, String suffix) {
        for (Pair<String, Parameter> p : block.getParameters()) {
            if (p.getKey().equalsIgnoreCase(suffix)) {
                return p.getValue().getArray();
            }
        }
        throw new IllegalStateException(
                suffix + " 없음: " + block.getParameters().keys());
    }

    private static String kindOf(Block block) {
        if (block instanceof LambdaBlock lb) {
            return lb.getName();
        }
        return block.getClass().getSimpleName();
    }

    private static String labelOf(Block block, String kind, NDArray out) {
        if ("Conv2d".equals(kind)) {
            long[] w = block.getParameters().get(0).getValue().getArray().getShape().getShape();
            return String.format("Conv %d×%d → %d채널", w[2], w[3], w[0]);
        }
        if ("BatchNorm".equals(kind)) {
            return "BatchNorm (" + out.getShape().get(1) + "채널)";
        }
        if ("Linear".equals(kind)) {
            return "Linear → " + out.getShape().get(1) + "클래스";
        }
        return switch (kind) {
            case "ReLU" -> "ReLU";
            case "identity" -> "항등 (그대로 통과)";
            case "globalAvgPool2d" -> "전역 평균 풀링";
            case "batchFlatten" -> "펼치기";
            default -> kind;
        };
    }

    /** 곱셈-누산 횟수. 가중치가 있는 Conv와 Linear만 의미가 있다. */
    private static long macs(Block block, String kind, NDArray in, NDArray out) {
        long[] o = out.getShape().getShape();
        long[] i = in.getShape().getShape();
        if ("Conv2d".equals(kind) && o.length == 4) {
            long[] w = block.getParameters().get(0).getValue().getArray().getShape().getShape();
            return o[1] * o[2] * o[3] * i[1] * w[2] * w[3];
        }
        if ("Linear".equals(kind)) {
            return o[1] * i[1];
        }
        return 0;
    }

    private static Map<String, Object> stats(NDArray a) {
        return stats(a.toFloatArray());
    }

    private static Map<String, Object> stats(float[] v) {
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        double sum = 0;
        int zeros = 0;
        for (float f : v) {
            min = Math.min(min, f);
            max = Math.max(max, f);
            sum += f;
            if (f == 0f) {
                zeros++;
            }
        }
        double mean = sum / v.length;
        double sq = 0;
        for (float f : v) {
            sq += (f - mean) * (f - mean);
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("n", v.length);
        m.put("min", min);
        m.put("max", max);
        m.put("mean", mean);
        m.put("std", Math.sqrt(sq / v.length));
        m.put("zeroFrac", (double) zeros / v.length);
        return m;
    }

    private static List<Map<String, Object>> maps(NDArray a) {
        List<Map<String, Object>> list = new ArrayList<>();
        long channels = a.getShape().get(1);
        for (int c = 0; c < Math.min(MAX_MAPS, channels); c++) {
            Map<String, Object> m = plane(a.get(new NDIndex(0, c)));
            m.put("c", c);
            list.add(m);
        }
        return list;
    }

    /** (H, W) 평면 하나를 0~255 그레이스케일로 눌러 base64로 싣는다. */
    private static Map<String, Object> plane(NDArray a) {
        long[] s = a.getShape().getShape();
        float[] v = a.toFloatArray();
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (float f : v) {
            min = Math.min(min, f);
            max = Math.max(max, f);
        }
        float range = max - min;
        byte[] px = new byte[v.length];
        for (int i = 0; i < v.length; i++) {
            px[i] = (byte) (range == 0 ? 0 : Math.round((v[i] - min) / range * 255f));
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("h", s[0]);
        m.put("w", s[1]);
        m.put("min", min);
        m.put("max", max);
        m.put("b64", Base64.getEncoder().encodeToString(px));
        return m;
    }

    private static double norm(float[] v) {
        double s = 0;
        for (float f : v) {
            s += (double) f * f;
        }
        return Math.sqrt(s);
    }

    private static double absMean(float[] v) {
        double s = 0;
        for (float f : v) {
            s += Math.abs(f);
        }
        return s / v.length;
    }

    private static double absMax(float[] v) {
        double s = 0;
        for (float f : v) {
            s = Math.max(s, Math.abs(f));
        }
        return s;
    }

    private static double[] floats(float[] v) {
        double[] d = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            d[i] = v[i];
        }
        return d;
    }

    private static long[] out(Shape s) {
        return s.getShape();
    }

    private static Map<String, Object> specInfo(ResNetSpec spec) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("bottleneck", spec.bottleneck());
        m.put("units", spec.units());
        m.put("stageFilters", spec.stageFilters());
        m.put("stemFilters", spec.stemFilters());
        return m;
    }

    private static int labelOf(Path path) {
        String dir = path.getParent().getFileName().toString();
        for (int i = 0; i < CLASSES.length; i++) {
            if (CLASSES[i].equals(dir)) {
                return i;
            }
        }
        throw new IllegalArgumentException(
                "이미지 경로의 상위 폴더가 CIFAR-10 클래스 이름이어야 한다: " + dir);
    }

    private static void console(
            String version,
            Map<String, Object> root,
            int label,
            NDArray logits,
            float loss1,
            NDArray logits2,
            float loss2) {
        System.out.println();
        System.out.println("=".repeat(74));
        System.out.printf("  ResNet %s — 이미지 한 장, 사이클 한 번%n", version);
        System.out.println("=".repeat(74));
        System.out.printf("정답: %s (%d)%n", CLASSES[label], label);
        System.out.printf("기록한 연산 단계: %d개%n", ((List<?>) root.get("steps")).size());
        System.out.printf("로짓(전) : %s%n", Arrays.toString(logits.toFloatArray()));
        System.out.printf("로짓(후) : %s%n", Arrays.toString(logits2.toFloatArray()));
        System.out.printf("손실 %.6f -> %.6f  (%+.6f)%n", loss1, loss2, loss2 - loss1);
    }

    // ────────────────────────────────────────────────────────────────────────────

    private record Opt(
            List<String> versions, int depth, String image, float lr, long seed, String outDir) {

        static Opt parse(String[] args) {
            List<String> versions = List.of("v1", "v2");
            int depth = 20;
            String image = "assets/cifar10/test/cat/test_01124.png";
            float lr = 0.1f;
            long seed = 42;
            String outDir = "build/probe";
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--version" -> versions = List.of(args[++i].split(","));
                    case "--depth" -> depth = Integer.parseInt(args[++i]);
                    case "--image" -> image = args[++i];
                    case "--lr" -> lr = Float.parseFloat(args[++i]);
                    case "--seed" -> seed = Long.parseLong(args[++i]);
                    case "--out" -> outDir = args[++i];
                    default -> throw new IllegalArgumentException("알 수 없는 옵션: " + args[i]);
                }
            }
            return new Opt(versions, depth, image, lr, seed, outDir);
        }
    }

    /** 의존성을 늘리지 않으려고 직접 쓴 최소 JSON 직렬화기. */
    private static final class Json {

        private static final MathContext MC = new MathContext(7);

        static String write(Object o) {
            StringBuilder sb = new StringBuilder(1 << 20);
            value(sb, o);
            return sb.toString();
        }

        private static void value(StringBuilder sb, Object o) {
            switch (o) {
                case null -> sb.append("null");
                case Map<?, ?> m -> {
                    sb.append('{');
                    boolean first = true;
                    for (Map.Entry<?, ?> e : m.entrySet()) {
                        if (!first) {
                            sb.append(',');
                        }
                        first = false;
                        string(sb, String.valueOf(e.getKey()));
                        sb.append(':');
                        value(sb, e.getValue());
                    }
                    sb.append('}');
                }
                case Iterable<?> it -> {
                    sb.append('[');
                    boolean first = true;
                    for (Object e : it) {
                        if (!first) {
                            sb.append(',');
                        }
                        first = false;
                        value(sb, e);
                    }
                    sb.append(']');
                }
                case String s -> string(sb, s);
                case Boolean b -> sb.append(b);
                case Number n -> number(sb, n.doubleValue());
                case double[] a -> array(sb, a);
                case float[] a -> {
                    sb.append('[');
                    for (int i = 0; i < a.length; i++) {
                        if (i > 0) {
                            sb.append(',');
                        }
                        number(sb, a[i]);
                    }
                    sb.append(']');
                }
                case long[] a -> {
                    sb.append('[');
                    for (int i = 0; i < a.length; i++) {
                        if (i > 0) {
                            sb.append(',');
                        }
                        sb.append(a[i]);
                    }
                    sb.append(']');
                }
                case int[] a -> {
                    sb.append('[');
                    for (int i = 0; i < a.length; i++) {
                        if (i > 0) {
                            sb.append(',');
                        }
                        sb.append(a[i]);
                    }
                    sb.append(']');
                }
                case Object[] a -> {
                    sb.append('[');
                    for (int i = 0; i < a.length; i++) {
                        if (i > 0) {
                            sb.append(',');
                        }
                        value(sb, a[i]);
                    }
                    sb.append(']');
                }
                default -> string(sb, String.valueOf(o));
            }
        }

        private static void array(StringBuilder sb, double[] a) {
            sb.append('[');
            for (int i = 0; i < a.length; i++) {
                if (i > 0) {
                    sb.append(',');
                }
                number(sb, a[i]);
            }
            sb.append(']');
        }

        private static void number(StringBuilder sb, double d) {
            if (!Double.isFinite(d)) {
                sb.append("null");
            } else if (d == 0) {
                sb.append('0');
            } else if (d == Math.rint(d) && Math.abs(d) < 1e15) {
                sb.append((long) d);
            } else {
                sb.append(new BigDecimal(d).round(MC).stripTrailingZeros().toString());
            }
        }

        private static void string(StringBuilder sb, String s) {
            sb.append('"');
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                switch (c) {
                    case '"' -> sb.append("\\\"");
                    case '\\' -> sb.append("\\\\");
                    case '\n' -> sb.append("\\n");
                    case '\r' -> sb.append("\\r");
                    case '\t' -> sb.append("\\t");
                    default -> {
                        if (c < 0x20) {
                            sb.append(String.format("\\u%04x", (int) c));
                        } else {
                            sb.append(c);
                        }
                    }
                }
            }
            sb.append('"');
        }
    }
}
