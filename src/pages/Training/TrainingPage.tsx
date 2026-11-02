import React from 'react';
import {
  Play,
  Square,
  RotateCcw,
  Activity,
  TrendingDown,
  Target,
  Clock,
  LineChart,
} from 'lucide-react';
import { NetworkVisualizer } from '../../components/NeuralNetwork/NetworkVisualizer';
import { DatasetPlot } from '../../components/Charts/DatasetPlot';
import { StatCard } from '../../components/Statistics/StatCard';
import { DATASET_LIST } from '../../services/datasets';
import {
  DatasetId,
  GeneratedDataset,
  NetworkArchitecture,
  NetworkSnapshot,
  OptimizerType,
  PageId,
  PredictionVisualData,
  TrainingConfig,
  TrainingStatus,
} from '../../types';
import {
  formatArchitectureString,
  formatMetric,
  formatPercent,
  getOptimizerLabel,
  getTrainingStatusLabel,
} from '../../utils/formatters';
import './TrainingPage.css';

interface TrainingPageProps {
  architecture: NetworkArchitecture;
  snapshot: NetworkSnapshot;
  dataset: GeneratedDataset;
  predictionVisuals: PredictionVisualData | null;
  trainingConfig: TrainingConfig;
  trainingStatus: TrainingStatus;
  currentEpoch: number;
  currentLoss: number | null;
  currentAccuracy: number | null;
  elapsedMs: number;
  onSelectDataset: (id: DatasetId) => void;
  onUpdateTrainingConfig: (patch: Partial<TrainingConfig>) => void;
  onStartTraining: () => void;
  onStopTraining: () => void;
  onResetModel: () => void;
  onProbeCoordinate: (inputVec: number[]) => void;
  onTriggerPulse: () => void;
  onNavigate: (page: PageId) => void;
}

const LEARNING_RATES = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3];
const BATCH_SIZES = [16, 32, 64, 128];
const OPTIMIZERS: OptimizerType[] = ['adam', 'sgd', 'rmsprop'];

export const TrainingPage: React.FC<TrainingPageProps> = ({
  architecture,
  snapshot,
  dataset,
  predictionVisuals,
  trainingConfig,
  trainingStatus,
  currentEpoch,
  currentLoss,
  currentAccuracy,
  elapsedMs,
  onSelectDataset,
  onUpdateTrainingConfig,
  onStartTraining,
  onStopTraining,
  onResetModel,
  onProbeCoordinate,
  onTriggerPulse,
  onNavigate,
}) => {
  const isTraining = trainingStatus === 'training';
  const progressPercent =
    trainingConfig.epochs > 0
      ? Math.min(100, Math.round((currentEpoch / trainingConfig.epochs) * 100))
      : 0;

  return (
    <div className="np-training-page">
      <div className="page-header-bar">
        <div>
          <h1 className="page-title">04 // ПУЛЬТ УПРАВЛЕНИЯ ОБУЧЕНИЕМ (TENSORFLOW.JS)</h1>
          <p className="page-subtitle">
            Настройка гиперпараметров оптимизации и наблюдение за схождением весов в реальном времени
          </p>
        </div>

        <div className="np-trn-header-btns">
          <button
            type="button"
            className="retro-btn"
            onClick={() => onNavigate('analytics')}
          >
            <LineChart size={14} />
            ОТКРЫТЬ АНАЛИТИКУ
          </button>
          <button
            type="button"
            className="retro-btn"
            onClick={onResetModel}
            disabled={isTraining}
          >
            <RotateCcw size={14} />
            СБРОСИТЬ МОДЕЛЬ
          </button>
          {isTraining ? (
            <button
              type="button"
              className="retro-btn retro-btn-danger"
              onClick={onStopTraining}
            >
              <Square size={14} />
              ОСТАНОВИТЬ
            </button>
          ) : (
            <button
              type="button"
              className="retro-btn retro-btn-primary"
              onClick={onStartTraining}
            >
              <Play size={14} />
              ЗАПУСТИТЬ ОБУЧЕНИЕ
            </button>
          )}
        </div>
      </div>

      {/* Телеметрия текущего процесса обучения */}
      <div className="np-stat-grid">
        <StatCard
          code="EPC.01"
          label="Текущая эпоха"
          value={`${currentEpoch} / ${trainingConfig.epochs}`}
          subtext={`Выполнено ${progressPercent}% цикла`}
          accent={isTraining}
          icon={<Clock size={15} />}
        />
        <StatCard
          code="LSS.02"
          label="Ошибка (Loss)"
          value={formatMetric(currentLoss, 5)}
          subtext={
            dataset.meta.taskType === 'classification' &&
            architecture.outputActivation === 'sigmoid'
              ? 'Функция: Binary Crossentropy'
              : 'Функция: Mean Squared Error'
          }
          icon={<TrendingDown size={15} />}
        />
        <StatCard
          code="ACC.03"
          label={
            dataset.meta.taskType === 'classification'
              ? 'Точность (Accuracy)'
              : 'Точность аппрокс. (R²)'
          }
          value={formatPercent(currentAccuracy, 1)}
          subtext={`Выборка: ${dataset.meta.code} (${dataset.points.length} точек)`}
          accent
          icon={<Target size={15} />}
        />
        <StatCard
          code="STA.04"
          label="Состояние ядра"
          value={getTrainingStatusLabel(trainingStatus)}
          subtext={
            elapsedMs > 0
              ? `Время вычислений: ${(elapsedMs / 1000).toFixed(2)} сек`
              : `Архитектура: ${formatArchitectureString(architecture)}`
          }
          icon={<Activity size={15} />}
        />
      </div>

      {/* Полоса прогресса и панель гиперпараметров */}
      <div className="tech-panel np-trn-controls-panel">
        <div className="np-trn-progress-row">
          <div className="np-trn-progress-label">
            <span>ПРОГРЕСС ОБУЧЕНИЯ СЕТИ</span>
            <strong className="phosphor-text">{progressPercent}%</strong>
          </div>
          <div className="np-trn-progress-track">
            <div
              className={`np-trn-progress-fill ${isTraining ? 'is-animating' : ''}`}
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>

        <div className="np-trn-settings-grid">
          <div className="np-trn-field">
            <label>Набор данных эксперимента</label>
            <select
              className="retro-select"
              value={dataset.meta.id}
              disabled={isTraining}
              onChange={(e) => onSelectDataset(e.target.value as DatasetId)}
            >
              {DATASET_LIST.map((ds) => (
                <option key={ds.id} value={ds.id}>
                  {ds.name} [{ds.code}]
                </option>
              ))}
            </select>
          </div>

          <div className="np-trn-field">
            <label>
              Количество эпох: <strong className="phosphor-text">{trainingConfig.epochs}</strong>
            </label>
            <div className="np-trn-epoch-controls">
              <input
                type="range"
                className="retro-range"
                min={10}
                max={400}
                step={10}
                value={trainingConfig.epochs}
                disabled={isTraining}
                onChange={(e) =>
                  onUpdateTrainingConfig({ epochs: Number(e.target.value) })
                }
              />
              <input
                type="number"
                className="retro-input np-epoch-num"
                min={5}
                max={1000}
                value={trainingConfig.epochs}
                disabled={isTraining}
                onChange={(e) => {
                  const val = Math.max(5, Math.min(1000, Number(e.target.value) || 10));
                  onUpdateTrainingConfig({ epochs: val });
                }}
              />
            </div>
          </div>

          <div className="np-trn-field">
            <label>Скорость обучения (Learning Rate)</label>
            <select
              className="retro-select"
              value={trainingConfig.learningRate}
              disabled={isTraining}
              onChange={(e) =>
                onUpdateTrainingConfig({ learningRate: Number(e.target.value) })
              }
            >
              {LEARNING_RATES.map((lr) => (
                <option key={lr} value={lr}>
                  LR = {lr}
                </option>
              ))}
            </select>
          </div>

          <div className="np-trn-field">
            <label>Алгоритм оптимизации</label>
            <select
              className="retro-select"
              value={trainingConfig.optimizer}
              disabled={isTraining}
              onChange={(e) =>
                onUpdateTrainingConfig({
                  optimizer: e.target.value as OptimizerType,
                })
              }
            >
              {OPTIMIZERS.map((opt) => (
                <option key={opt} value={opt}>
                  {getOptimizerLabel(opt)}
                </option>
              ))}
            </select>
          </div>

          <div className="np-trn-field">
            <label>Размер пакета (Batch Size)</label>
            <select
              className="retro-select"
              value={trainingConfig.batchSize}
              disabled={isTraining}
              onChange={(e) =>
                onUpdateTrainingConfig({ batchSize: Number(e.target.value) })
              }
            >
              {BATCH_SIZES.map((bs) => (
                <option key={bs} value={bs}>
                  {bs} примеров
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Синхронная визуализация графа нейросети и разделяющей поверхности */}
      <div className="np-trn-visuals-grid">
        <NetworkVisualizer
          snapshot={snapshot}
          isTraining={isTraining}
          height={390}
          onTriggerPulse={onTriggerPulse}
          title="ДИНАМИКА ВЕСОВ И АКТИВАЦИЙ В ПРОЦЕССЕ ОБУЧЕНИЯ"
        />

        <DatasetPlot
          meta={dataset.meta}
          points={dataset.points}
          predictionVisuals={predictionVisuals}
          height={390}
          onProbeCoordinate={onProbeCoordinate}
          title="ФОРМИРОВАНИЕ ГРАНИЦЫ РЕШЕНИЙ / АППРОКСИМАЦИИ"
        />
      </div>
    </div>
  );
};
