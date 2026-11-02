export type PageId = 'dashboard' | 'network' | 'datasets' | 'training' | 'analytics';

export type ActivationType = 'relu' | 'tanh' | 'sigmoid' | 'elu' | 'linear';

export type OptimizerType = 'adam' | 'sgd' | 'rmsprop';

export type DatasetId = 'xor' | 'circles' | 'linear' | 'clusters';

export type TaskType = 'classification' | 'regression';

export type TrainingStatus = 'idle' | 'training' | 'stopped' | 'completed' | 'error';

export type LogCategory = 'СИСТЕМА' | 'МОДЕЛЬ' | 'ДАННЫЕ' | 'ОБУЧЕНИЕ' | 'ОШИБКА' | 'КОМАНДА';

export interface HiddenLayerConfig {
  id: string;
  neurons: number;
  activation: ActivationType;
}

export interface NetworkArchitecture {
  inputNeurons: number;
  hiddenLayers: HiddenLayerConfig[];
  outputNeurons: number;
  outputActivation: ActivationType;
  autoOutputActivation: boolean;
}

export interface NetworkSnapshot {
  layerSizes: number[];
  layerLabels: string[];
  layerActivations: string[];
  /** weights[layerTransitionIdx][fromNeuronIdx][toNeuronIdx] */
  weights: number[][][];
  /** biases[denseLayerIdx][neuronIdx] */
  biases: number[][];
  /** activations[allLayerIdx][neuronIdx] (includes input layer at index 0) */
  activations: number[][];
  probeInput: number[];
  probeOutput: number[];
}

export interface DataPoint {
  x: number;
  y: number;
  label: number;
}

export interface DatasetMetadata {
  id: DatasetId;
  name: string;
  code: string;
  taskType: TaskType;
  inputDim: 1 | 2;
  description: string;
  difficulty: 'БАЗОВАЯ' | 'СРЕДНЯЯ' | 'ВЫСОКАЯ';
  recommendedHint: string;
}

export interface DatasetConfig {
  id: DatasetId;
  sampleCount: number;
  noise: number;
}

export interface GeneratedDataset {
  meta: DatasetMetadata;
  config: DatasetConfig;
  points: DataPoint[];
}

export interface PredictionVisualData {
  taskType: TaskType;
  /** For 2D classification: gridSize x gridSize matrix of predicted probabilities [0..1] */
  gridSize?: number;
  gridValues?: number[];
  /** For 1D->1D regression: sorted curve points (x, predictedY) */
  regressionCurve?: Array<{ x: number; y: number }>;
}

export interface TrainingConfig {
  epochs: number;
  learningRate: number;
  batchSize: number;
  optimizer: OptimizerType;
}

export interface EpochMetric {
  epoch: number;
  loss: number;
  accuracy: number;
  elapsedMs: number;
}

export interface TrainingRunSummary {
  id: string;
  runNumber: number;
  startedAt: string;
  datasetId: DatasetId;
  datasetName: string;
  taskType: TaskType;
  architectureSummary: string;
  totalParams: number;
  epochsCompleted: number;
  targetEpochs: number;
  learningRate: number;
  optimizer: OptimizerType;
  initialLoss: number;
  finalLoss: number;
  initialAccuracy: number;
  finalAccuracy: number;
  durationMs: number;
  status: 'completed' | 'stopped';
  history: EpochMetric[];
}

export interface TerminalLogEntry {
  id: string;
  timestamp: string;
  category: LogCategory;
  message: string;
  isError?: boolean;
}
