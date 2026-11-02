import * as tf from '@tensorflow/tfjs';
import {
  DataPoint,
  EpochMetric,
  NetworkArchitecture,
  TaskType,
  TrainingConfig,
} from '../types';
import { compileModelForTraining } from './neuralNetwork';

export interface TrainingSessionCallbacks {
  onEpochUpdate: (metric: EpochMetric) => void;
  shouldStop: () => boolean;
}

export interface TrainingSessionResult {
  completedEpochs: number;
  stoppedEarly: boolean;
  history: EpochMetric[];
  durationMs: number;
}

export async function runModelTraining(
  model: tf.Sequential,
  architecture: NetworkArchitecture,
  points: DataPoint[],
  taskType: TaskType,
  inputDim: 1 | 2,
  config: TrainingConfig,
  callbacks: TrainingSessionCallbacks
): Promise<TrainingSessionResult> {
  if (points.length === 0) {
    throw new Error('Набор данных пуст. Невозможно запустить обучение.');
  }

  // Перекомпиляция с актуальным оптимизатором и скоростью обучения, сохраняя текущие веса
  compileModelForTraining(model, architecture, config, taskType);

  const inputs =
    inputDim === 2 ? points.map((p) => [p.x, p.y]) : points.map((p) => [p.x]);
  const targets = points.map((p) => [taskType === 'classification' ? p.label : p.y]);

  const xs = tf.tensor2d(inputs, [points.length, inputDim]);
  const ys = tf.tensor2d(targets, [points.length, 1]);

  const history: EpochMetric[] = [];
  const startTime = performance.now();
  let stoppedEarly = false;

  try {
    await model.fit(xs, ys, {
      epochs: config.epochs,
      batchSize: Math.min(config.batchSize, points.length),
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          const rawLoss = typeof logs?.loss === 'number' && !Number.isNaN(logs.loss) ? logs.loss : 0;

          // Вычисляем реальную точность на текущих весах модели через тензоры TF.js
          const accuracy = tf.tidy(() => {
            const preds = model.predict(xs) as tf.Tensor;
            if (taskType === 'classification') {
              const predClasses = preds.greaterEqual(0.5);
              const trueClasses = ys.greaterEqual(0.5);
              return predClasses.equal(trueClasses).cast('float32').mean().dataSync()[0];
            } else {
              const ssRes = ys.sub(preds).square().sum().dataSync()[0];
              const ssTot = ys.sub(ys.mean()).square().sum().dataSync()[0];
              return ssTot > 1e-8 ? Math.max(0, Math.min(1, 1 - ssRes / ssTot)) : 0;
            }
          });

          const metric: EpochMetric = {
            epoch: epoch + 1,
            loss: rawLoss,
            accuracy,
            elapsedMs: Math.round(performance.now() - startTime),
          };

          history.push(metric);
          callbacks.onEpochUpdate(metric);

          if (callbacks.shouldStop()) {
            stoppedEarly = true;
            model.stopTraining = true;
          }

          // Уступаем кадр браузеру для плавной отрисовки графиков и анимации весов
          await tf.nextFrame();
        },
      },
    });
  } finally {
    xs.dispose();
    ys.dispose();
  }

  return {
    completedEpochs: history.length,
    stoppedEarly,
    history,
    durationMs: Math.round(performance.now() - startTime),
  };
}
