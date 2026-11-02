import {
  DataPoint,
  DatasetConfig,
  DatasetId,
  DatasetMetadata,
  GeneratedDataset,
} from '../types';

export const DATASETS_CATALOG: Record<DatasetId, DatasetMetadata> = {
  xor: {
    id: 'xor',
    name: 'XOR (Исключающее ИЛИ)',
    code: 'XOR-2D',
    taskType: 'classification',
    inputDim: 2,
    difficulty: 'СРЕДНЯЯ',
    description:
      'Классическая линейно неразделимая задача: четыре квадранта на плоскости с чередующимися классами 0 и 1.',
    recommendedHint:
      'Требуется минимум 1 скрытый слой (4–6 нейронов с активацией Tanh или ReLU). Линейный перцептрон без скрытых слоёв не способен решить XOR.',
  },
  circles: {
    id: 'circles',
    name: 'Классификация точек',
    code: 'CIRC-2D',
    taskType: 'classification',
    inputDim: 2,
    difficulty: 'ВЫСОКАЯ',
    description:
      'Концентрические кольцевые области точек двух классов: внутреннее ядро (класс 1) окружено внешним кольцом (класс 0).',
    recommendedHint:
      'Рекомендуется 1–2 скрытых слоя (например, 6 → 4 нейронов с Tanh или ReLU) для построения замкнутой нелинейной границы решений.',
  },
  linear: {
    id: 'linear',
    name: 'Линейная зависимость',
    code: 'LIN-REG',
    taskType: 'regression',
    inputDim: 1,
    difficulty: 'БАЗОВАЯ',
    description:
      'Непрерывная регрессионная выборка точек вдоль прямой Y = 0.68·X + 0.12 со случайным гауссовым шумом.',
    recommendedHint:
      'Задача аппроксимации функции: выходная активация Linear, функция потерь MSE. Даже сеть с 0–1 скрытым слоем быстро находит наклон и смещение.',
  },
  clusters: {
    id: 'clusters',
    name: 'Две группы точек',
    code: 'CLUST-2D',
    taskType: 'classification',
    inputDim: 2,
    difficulty: 'БАЗОВАЯ',
    description:
      'Два компактных гауссовых облака точек в противоположных областях плоскости признаков (X₁, X₂).',
    recommendedHint:
      'Линейно разделимая выборка: решается за считанные эпохи даже простейшим перцептроном без скрытых слоёв.',
  },
};

export const DATASET_LIST: DatasetMetadata[] = [
  DATASETS_CATALOG.xor,
  DATASETS_CATALOG.circles,
  DATASETS_CATALOG.linear,
  DATASETS_CATALOG.clusters,
];

function gaussianRandom(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function clampCoord(v: number): number {
  return Math.max(-1.05, Math.min(1.05, v));
}

export function generateDataset(config: DatasetConfig): GeneratedDataset {
  const meta = DATASETS_CATALOG[config.id];
  const count = Math.max(20, Math.min(500, Math.round(config.sampleCount)));
  const noise = Math.max(0, Math.min(0.4, config.noise));
  const points: DataPoint[] = [];

  if (config.id === 'xor') {
    for (let i = 0; i < count; i++) {
      const qx = i % 2 === 0 ? 0.52 : -0.52;
      const qy = i % 4 < 2 ? 0.52 : -0.52;
      const spread = 0.16 + noise * 0.65;
      const x = clampCoord(qx + gaussianRandom() * spread);
      const y = clampCoord(qy + gaussianRandom() * spread);
      const label = (qx > 0 && qy > 0) || (qx < 0 && qy < 0) ? 1 : 0;
      points.push({ x, y, label });
    }
  } else if (config.id === 'circles') {
    for (let i = 0; i < count; i++) {
      const isInner = i % 2 === 0;
      const angle = Math.random() * Math.PI * 2;
      const baseRadius = isInner
        ? Math.random() * 0.34
        : 0.68 + Math.random() * 0.22;
      const r = Math.max(0.02, baseRadius + gaussianRandom() * (noise * 0.6));
      const x = clampCoord(Math.cos(angle) * r);
      const y = clampCoord(Math.sin(angle) * r);
      points.push({ x, y, label: isInner ? 1 : 0 });
    }
  } else if (config.id === 'linear') {
    for (let i = 0; i < count; i++) {
      const x = -0.92 + (i / Math.max(1, count - 1)) * 1.84;
      const jitterX = clampCoord(x + gaussianRandom() * 0.02);
      const cleanY = 0.68 * jitterX + 0.12;
      const y = clampCoord(cleanY + gaussianRandom() * (0.04 + noise * 0.75));
      points.push({ x: jitterX, y, label: y });
    }
  } else {
    // clusters — Две группы точек
    for (let i = 0; i < count; i++) {
      const cls = i % 2 === 0 ? 1 : 0;
      const cx = cls === 1 ? 0.5 : -0.5;
      const cy = cls === 1 ? 0.46 : -0.46;
      const spread = 0.18 + noise * 0.7;
      const x = clampCoord(cx + gaussianRandom() * spread);
      const y = clampCoord(cy + gaussianRandom() * spread);
      points.push({ x, y, label: cls });
    }
  }

  return {
    meta,
    config: { ...config, sampleCount: count, noise },
    points,
  };
}
