import { ActivationType, NetworkArchitecture, OptimizerType, TrainingStatus } from '../types';

export function formatTimestamp(date: Date = new Date()): string {
  const hh = String(date.getHours()).padStart(2, '0');
  const mm = String(date.getMinutes()).padStart(2, '0');
  const ss = String(date.getSeconds()).padStart(2, '0');
  const ms = String(date.getMilliseconds()).padStart(3, '0');
  return `${hh}:${mm}:${ss}.${ms}`;
}

export function calculateLayerSizes(arch: NetworkArchitecture): number[] {
  return [
    arch.inputNeurons,
    ...arch.hiddenLayers.map((l) => l.neurons),
    arch.outputNeurons,
  ];
}

export function calculateTotalNeurons(arch: NetworkArchitecture): number {
  return calculateLayerSizes(arch).reduce((sum, count) => sum + count, 0);
}

export function calculateTotalLayers(arch: NetworkArchitecture): number {
  return 2 + arch.hiddenLayers.length;
}

export function calculateTotalParameters(arch: NetworkArchitecture): number {
  const sizes = calculateLayerSizes(arch);
  let total = 0;
  for (let i = 1; i < sizes.length; i++) {
    const weights = sizes[i - 1] * sizes[i];
    const biases = sizes[i];
    total += weights + biases;
  }
  return total;
}

export function formatArchitectureString(arch: NetworkArchitecture): string {
  const parts: string[] = [`${arch.inputNeurons} [ВХОД]`];
  arch.hiddenLayers.forEach((l) => {
    parts.push(`${l.neurons} (${l.activation.toUpperCase()})`);
  });
  parts.push(`${arch.outputNeurons} (${arch.outputActivation.toUpperCase()})`);
  return parts.join(' → ');
}

export function getActivationLabel(act: ActivationType): string {
  switch (act) {
    case 'relu':
      return 'ReLU — Линейный выпрямитель';
    case 'tanh':
      return 'Tanh — Гиперболический тангенс';
    case 'sigmoid':
      return 'Sigmoid — Логистическая сигмоида';
    case 'elu':
      return 'ELU — Экспоненциальный блок';
    case 'linear':
      return 'Linear — Линейная тождественная';
  }
}

export function getOptimizerLabel(opt: OptimizerType): string {
  switch (opt) {
    case 'adam':
      return 'Adam (Адаптивный момент)';
    case 'sgd':
      return 'SGD (Стохастический градиент)';
    case 'rmsprop':
      return 'RMSprop (Среднеквадратичный градиент)';
  }
}

export function getTrainingStatusLabel(status: TrainingStatus): string {
  switch (status) {
    case 'idle':
      return 'ОЖИДАНИЕ';
    case 'training':
      return 'ОБУЧЕНИЕ АКТИВНО';
    case 'stopped':
      return 'ОСТАНОВЛЕНО';
    case 'completed':
      return 'ОБУЧЕНИЕ ЗАВЕРШЕНО';
    case 'error':
      return 'ОШИБКА ВЫЧИСЛЕНИЙ';
  }
}

export function formatMetric(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—';
  }
  return value.toFixed(digits);
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—';
  }
  return `${(value * 100).toFixed(digits)}%`;
}
