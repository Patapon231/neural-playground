import * as tf from '@tensorflow/tfjs';
import {
  DataPoint,
  NetworkArchitecture,
  NetworkSnapshot,
  OptimizerType,
  PredictionVisualData,
  TaskType,
  TrainingConfig,
} from '../types';
import { calculateLayerSizes } from '../utils/formatters';

export interface InitializedEngine {
  backend: string;
  tensorCount: number;
}

export async function initTensorFlowEngine(): Promise<InitializedEngine> {
  await tf.ready();
  let backend = tf.getBackend();
  if (!backend) {
    try {
      await tf.setBackend('webgl');
    } catch {
      await tf.setBackend('cpu');
    }
    backend = tf.getBackend() || 'cpu';
  }
  return {
    backend,
    tensorCount: tf.memory().numTensors,
  };
}

export function getTensorMemoryStats(): { numTensors: number; numBytes: number } {
  const mem = tf.memory();
  return {
    numTensors: mem.numTensors,
    numBytes: mem.numBytes,
  };
}

function createOptimizer(type: OptimizerType, learningRate: number): tf.Optimizer {
  switch (type) {
    case 'adam':
      return tf.train.adam(learningRate);
    case 'sgd':
      return tf.train.sgd(learningRate);
    case 'rmsprop':
      return tf.train.rmsprop(learningRate);
  }
}

export function compileModelForTraining(
  model: tf.Sequential,
  architecture: NetworkArchitecture,
  trainingConfig: TrainingConfig,
  taskType: TaskType
): void {
  const optimizer = createOptimizer(trainingConfig.optimizer, trainingConfig.learningRate);
  const useBinaryCrossentropy =
    taskType === 'classification' && architecture.outputActivation === 'sigmoid';

  model.compile({
    optimizer,
    loss: useBinaryCrossentropy ? 'binaryCrossentropy' : 'meanSquaredError',
  });
}

export function buildSequentialModel(
  architecture: NetworkArchitecture,
  trainingConfig: TrainingConfig,
  taskType: TaskType
): { model: tf.Sequential; paramCount: number } {
  const model = tf.sequential();

  if (architecture.hiddenLayers.length === 0) {
    model.add(
      tf.layers.dense({
        inputShape: [architecture.inputNeurons],
        units: architecture.outputNeurons,
        activation: architecture.outputActivation,
        kernelInitializer: 'glorotNormal',
        biasInitializer: 'zeros',
      })
    );
  } else {
    architecture.hiddenLayers.forEach((layer, idx) => {
      if (idx === 0) {
        model.add(
          tf.layers.dense({
            inputShape: [architecture.inputNeurons],
            units: layer.neurons,
            activation: layer.activation,
            kernelInitializer: 'glorotNormal',
            biasInitializer: 'zeros',
          })
        );
      } else {
        model.add(
          tf.layers.dense({
            units: layer.neurons,
            activation: layer.activation,
            kernelInitializer: 'glorotNormal',
            biasInitializer: 'zeros',
          })
        );
      }
    });

    model.add(
      tf.layers.dense({
        units: architecture.outputNeurons,
        activation: architecture.outputActivation,
        kernelInitializer: 'glorotNormal',
        biasInitializer: 'zeros',
      })
    );
  }

  compileModelForTraining(model, architecture, trainingConfig, taskType);

  return {
    model,
    paramCount: model.countParams(),
  };
}

export function createFallbackSnapshot(architecture: NetworkArchitecture): NetworkSnapshot {
  const layerSizes = calculateLayerSizes(architecture);
  const layerLabels = [
    'ВХОД',
    ...architecture.hiddenLayers.map((_, idx) => `СКР. ${idx + 1}`),
    'ВЫХОД',
  ];
  const layerActivations = [
    'input',
    ...architecture.hiddenLayers.map((l) => l.activation),
    architecture.outputActivation,
  ];

  const weights: number[][][] = [];
  const biases: number[][] = [];
  for (let l = 0; l < layerSizes.length - 1; l++) {
    const fromCount = layerSizes[l];
    const toCount = layerSizes[l + 1];
    const mat: number[][] = [];
    for (let i = 0; i < fromCount; i++) {
      mat.push(new Array(toCount).fill(0));
    }
    weights.push(mat);
    biases.push(new Array(toCount).fill(0));
  }

  const probeInput = new Array(architecture.inputNeurons).fill(0.5);
  const activations = layerSizes.map((size, idx) =>
    idx === 0 ? [...probeInput] : new Array(size).fill(0)
  );

  return {
    layerSizes,
    layerLabels,
    layerActivations,
    weights,
    biases,
    activations,
    probeInput,
    probeOutput: [0],
  };
}

export function extractNetworkSnapshot(
  model: tf.Sequential,
  architecture: NetworkArchitecture,
  probeInput?: number[]
): NetworkSnapshot {
  const layerSizes = calculateLayerSizes(architecture);
  const layerLabels = [
    'ВХОД',
    ...architecture.hiddenLayers.map((_, idx) => `СКР. ${idx + 1}`),
    'ВЫХОД',
  ];
  const layerActivations = [
    'ВХОД',
    ...architecture.hiddenLayers.map((l) => l.activation),
    architecture.outputActivation,
  ];

  const inputVec =
    probeInput && probeInput.length === architecture.inputNeurons
      ? probeInput
      : architecture.inputNeurons === 2
        ? [0.65, -0.45]
        : [0.55];

  const weights: number[][][] = [];
  const biases: number[][] = [];

  for (const layer of model.layers) {
    const layerWeights = layer.getWeights();
    if (layerWeights.length >= 2) {
      const kernel = layerWeights[0].arraySync() as number[][];
      const bias = Array.from(layerWeights[1].dataSync());
      weights.push(kernel);
      biases.push(bias);
    } else if (layerWeights.length === 1) {
      const kernel = layerWeights[0].arraySync() as number[][];
      weights.push(kernel);
      biases.push(new Array(kernel[0]?.length ?? 1).fill(0));
    }
  }

  const activations: number[][] = [[...inputVec]];

  const forwardActivations = tf.tidy(() => {
    let current: tf.Tensor = tf.tensor2d([inputVec], [1, inputVec.length]);
    const collected: number[][] = [];
    for (const layer of model.layers) {
      current = layer.apply(current) as tf.Tensor;
      collected.push(Array.from(current.dataSync()));
    }
    return collected;
  });

  activations.push(...forwardActivations);
  const probeOutput = activations[activations.length - 1] ?? [0];

  return {
    layerSizes,
    layerLabels,
    layerActivations,
    weights,
    biases,
    activations,
    probeInput: inputVec,
    probeOutput,
  };
}

export function computePredictionVisuals(
  model: tf.Sequential,
  taskType: TaskType,
  inputDim: 1 | 2,
  gridSize = 28
): PredictionVisualData {
  return tf.tidy(() => {
    if (taskType === 'classification' && inputDim === 2) {
      const coords: number[][] = [];
      const min = -1.1;
      const max = 1.1;
      for (let row = 0; row < gridSize; row++) {
        // Сверху вниз по оси Y
        const y = max - (row / (gridSize - 1)) * (max - min);
        for (let col = 0; col < gridSize; col++) {
          const x = min + (col / (gridSize - 1)) * (max - min);
          coords.push([x, y]);
        }
      }
      const gridTensor = tf.tensor2d(coords, [coords.length, 2]);
      const predTensor = model.predict(gridTensor) as tf.Tensor;
      const rawValues = Array.from(predTensor.dataSync());
      // Нормализуем для отображения карты решений в диапазон [0..1]
      const clamped = rawValues.map((v) => Math.max(0, Math.min(1, v)));
      return {
        taskType,
        gridSize,
        gridValues: clamped,
      };
    }

    // Задача линейной регрессии (1D вход X -> выход Y)
    const steps = 64;
    const xs: number[] = [];
    for (let i = 0; i < steps; i++) {
      xs.push(-1.05 + (i / (steps - 1)) * 2.1);
    }
    const xTensor = tf.tensor2d(
      xs.map((x) => [x]),
      [steps, 1]
    );
    const yTensor = model.predict(xTensor) as tf.Tensor;
    const ys = Array.from(yTensor.dataSync());

    return {
      taskType,
      regressionCurve: xs.map((x, idx) => ({ x, y: ys[idx] })),
    };
  });
}

export function evaluateDatasetMetrics(
  model: tf.Sequential,
  points: DataPoint[],
  taskType: TaskType,
  inputDim: 1 | 2
): { loss: number; accuracy: number } {
  if (points.length === 0) {
    return { loss: 0, accuracy: 0 };
  }

  return tf.tidy(() => {
    const inputs =
      inputDim === 2 ? points.map((p) => [p.x, p.y]) : points.map((p) => [p.x]);
    const targets = points.map((p) => [taskType === 'classification' ? p.label : p.y]);

    const xs = tf.tensor2d(inputs, [points.length, inputDim]);
    const ys = tf.tensor2d(targets, [points.length, 1]);
    const preds = model.predict(xs) as tf.Tensor;

    const mse = ys.sub(preds).square().mean().dataSync()[0];

    if (taskType === 'classification') {
      const predClasses = preds.greaterEqual(0.5);
      const trueClasses = ys.greaterEqual(0.5);
      const acc = predClasses.equal(trueClasses).cast('float32').mean().dataSync()[0];
      return { loss: mse, accuracy: acc };
    } else {
      // Коэффициент детерминации R², ограниченный в [0, 1] как показатель точности регрессии
      const ssRes = ys.sub(preds).square().sum().dataSync()[0];
      const yMean = ys.mean();
      const ssTot = ys.sub(yMean).square().sum().dataSync()[0];
      const r2 = ssTot > 1e-8 ? Math.max(0, Math.min(1, 1 - ssRes / ssTot)) : 0;
      return { loss: mse, accuracy: r2 };
    }
  });
}
