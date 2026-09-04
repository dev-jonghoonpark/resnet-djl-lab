package lab.resnet;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Cifar10;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.util.cuda.CudaUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * CIFAR-10 학습 루프. v1과 v2를 같은 조건에서 돌려 비교하는 것이 목적이다.
 *
 * <pre>
 *   mvn exec:java -Dexec.mainClass=lab.resnet.Train -Dexec.args="--version v2 --depth 20"
 * </pre>
 *
 * <p>하이퍼파라미터는 논문(He et al. 2015, 4.2절)을 따른다 — SGD momentum 0.9,
 * weight decay 1e-4, 초기 lr 0.1, 전체 학습의 50%와 75% 지점에서 lr을 1/10로.
 * 가중치 초기화는 MSRA/He 방식이다.
 */
public final class Train {

    private Train() {}

    public static void main(String[] args) throws IOException, TranslateException {
        Options opt = Options.parse(args);
        System.out.println(opt);

        Block net =
                "v1".equals(opt.version)
                        ? ResNetV1.create(ResNetSpec.cifar(opt.depth), 10)
                        : ResNetV2.create(ResNetSpec.cifar(opt.depth), 10);

        Cifar10 trainSet = cifar(Dataset.Usage.TRAIN, opt, opt.augment);
        Cifar10 testSet = cifar(Dataset.Usage.TEST, opt, false);
        trainSet.prepare(new ai.djl.training.util.ProgressBar());
        testSet.prepare(new ai.djl.training.util.ProgressBar());

        int batchesPerEpoch = (int) Math.ceil((double) trainSet.size() / opt.batchSize);
        int totalUpdates = batchesPerEpoch * opt.epochs;
        System.out.printf(
                "학습 샘플 %d개, 에폭당 %d배치, 총 %d업데이트%n",
                trainSet.size(), batchesPerEpoch, totalUpdates);

        String name = "resnet-" + opt.version + "-" + opt.depth;
        Path outDir = Paths.get(opt.outDir).resolve(name);
        Files.createDirectories(outDir);

        try (Model model = Model.newInstance(name)) {
            model.setBlock(net);
            try (Trainer trainer = model.newTrainer(config(opt, totalUpdates, outDir))) {
                trainer.setMetrics(new Metrics());
                trainer.initialize(new Shape(opt.batchSize, 3, 32, 32));

                EasyTrain.fit(trainer, opt.epochs, trainSet, testSet);

                TrainingResult result = trainer.getTrainingResult();
                System.out.println();
                System.out.println("=".repeat(74));
                System.out.printf(
                        "  %s — 학습 정확도 %.4f / 검증 정확도 %.4f / 검증 손실 %.4f%n",
                        name,
                        result.getTrainEvaluation("Accuracy"),
                        result.getValidateEvaluation("Accuracy"),
                        result.getValidateLoss());
                System.out.println("=".repeat(74));
            }
            model.save(outDir, name);
            System.out.println("모델 저장: " + outDir);
        }
    }

    private static Cifar10 cifar(Dataset.Usage usage, Options opt, boolean augment) {
        Pipeline pipeline = new Pipeline();
        if (augment) {
            // 증강은 반드시 ToTensor 앞에 온다 — HWC [0,255] 상태에서 크롭/반전한다.
            pipeline.add(new CifarAugmentation(opt.seed));
        }
        pipeline.add(new ToTensor())
                .add(new Normalize(Cifar10.NORMALIZE_MEAN, Cifar10.NORMALIZE_STD));

        return Cifar10.builder()
                .optUsage(usage)
                .setSampling(opt.batchSize, usage == Dataset.Usage.TRAIN)
                .optPipeline(pipeline)
                .optLimit(opt.limit)
                .build();
    }

    private static DefaultTrainingConfig config(Options opt, int totalUpdates, Path outDir) {
        Tracker lr = learningRate(opt.learningRate, totalUpdates);
        Optimizer sgd =
                Optimizer.sgd()
                        .setLearningRateTracker(lr)
                        .optMomentum(0.9f)
                        .optWeightDecays(1e-4f)
                        .build();

        SaveModelTrainingListener checkpoint = new SaveModelTrainingListener(outDir.toString());
        checkpoint.setSaveModelCallback(
                trainer -> {
                    TrainingResult r = trainer.getTrainingResult();
                    Model m = trainer.getModel();
                    m.setProperty("Epoch", String.valueOf(r.getEpoch()));
                    m.setProperty(
                            "Accuracy", String.format("%.5f", r.getValidateEvaluation("Accuracy")));
                });

        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                // MSRA/He 초기화. ResNet 논문이 쓴 방식이고, 깊은 망에서 Xavier보다 낫다.
                .optInitializer(
                        new XavierInitializer(
                                XavierInitializer.RandomType.GAUSSIAN,
                                XavierInitializer.FactorType.IN,
                                2f),
                        Parameter.Type.WEIGHT)
                .optOptimizer(sgd)
                .optDevices(devices(opt.maxGpus))
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging())
                .addTrainingListeners(checkpoint);
    }

    /** 전체 업데이트의 50%, 75% 지점에서 lr을 1/10로 낮춘다 (논문의 32k/48k of 64k). */
    private static Tracker learningRate(float base, int totalUpdates) {
        int[] steps = {totalUpdates / 2, totalUpdates * 3 / 4};
        if (steps[0] <= 0 || steps[1] <= steps[0]) {
            // 스모크 런처럼 업데이트가 몇 안 되면 감쇠 없이 고정한다.
            return Tracker.fixed(base);
        }
        return Tracker.multiFactor().setBaseValue(base).setSteps(steps).optFactor(0.1f).build();
    }

    /** GPU가 있으면 쓰고 없으면 CPU로 떨어진다. */
    private static Device[] devices(int maxGpus) {
        int gpus = Math.min(CudaUtils.getGpuCount(), maxGpus);
        return gpus > 0
                ? Engine.getInstance().getDevices(gpus)
                : new Device[] {Device.cpu()};
    }

    /** 아주 작은 커맨드라인 파서. */
    private record Options(
            String version,
            int depth,
            int epochs,
            int batchSize,
            float learningRate,
            long limit,
            boolean augment,
            int maxGpus,
            long seed,
            String outDir) {

        static Options parse(String[] args) {
            String version = "v1";
            int depth = 20;
            int epochs = 2;
            int batchSize = 128;
            float lr = 0.1f;
            long limit = Long.MAX_VALUE;
            boolean augment = true;
            int maxGpus = 1;
            long seed = 42;
            String outDir = "build/model";

            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--version" -> version = require(args, ++i, "--version");
                    case "--depth" -> depth = Integer.parseInt(require(args, ++i, "--depth"));
                    case "--epochs" -> epochs = Integer.parseInt(require(args, ++i, "--epochs"));
                    case "--batch" -> batchSize = Integer.parseInt(require(args, ++i, "--batch"));
                    case "--lr" -> lr = Float.parseFloat(require(args, ++i, "--lr"));
                    case "--limit" -> limit = Long.parseLong(require(args, ++i, "--limit"));
                    case "--seed" -> seed = Long.parseLong(require(args, ++i, "--seed"));
                    case "--max-gpus" -> maxGpus = Integer.parseInt(require(args, ++i, "--max-gpus"));
                    case "--out" -> outDir = require(args, ++i, "--out");
                    case "--no-augment" -> augment = false;
                    default -> throw new IllegalArgumentException("알 수 없는 옵션: " + args[i]);
                }
            }
            if (!"v1".equals(version) && !"v2".equals(version)) {
                throw new IllegalArgumentException("--version은 v1 또는 v2여야 한다: " + version);
            }
            return new Options(
                    version, depth, epochs, batchSize, lr, limit, augment, maxGpus, seed, outDir);
        }

        private static String require(String[] args, int i, String flag) {
            if (i >= args.length) {
                throw new IllegalArgumentException(flag + "에 값이 필요하다");
            }
            return args[i];
        }

        @Override
        public String toString() {
            return String.format(
                    "ResNet %s-%d | 에폭 %d | 배치 %d | lr %.3f | 증강 %s | 샘플 제한 %s",
                    version,
                    depth,
                    epochs,
                    batchSize,
                    learningRate,
                    augment ? "on" : "off",
                    limit == Long.MAX_VALUE ? "없음" : String.valueOf(limit));
        }
    }
}
