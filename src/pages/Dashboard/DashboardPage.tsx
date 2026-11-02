import React from 'react';
import {
  Layers,
  Cpu,
  Sliders,
  Activity,
  ArrowRight,
  RotateCcw,
  Play,
  Database,
} from 'lucide-react';
import { StatCard } from '../../components/Statistics/StatCard';
import { NetworkVisualizer } from '../../components/NeuralNetwork/NetworkVisualizer';
import {
  GeneratedDataset,
  NetworkArchitecture,
  NetworkSnapshot,
  PageId,
  TerminalLogEntry,
  TrainingConfig,
  TrainingStatus,
} from '../../types';
import {
  calculateTotalLayers,
  calculateTotalNeurons,
  calculateTotalParameters,
  formatArchitectureString,
  formatMetric,
  formatPercent,
  getOptimizerLabel,
  getTrainingStatusLabel,
} from '../../utils/formatters';
import './DashboardPage.css';

interface DashboardPageProps {
  architecture: NetworkArchitecture;
  modelParamsCount: number;
  snapshot: NetworkSnapshot;
  dataset: GeneratedDataset;
  trainingConfig: TrainingConfig;
  trainingStatus: TrainingStatus;
  currentEpoch: number;
  currentLoss: number | null;
  currentAccuracy: number | null;
  logs: TerminalLogEntry[];
  onNavigate: (page: PageId) => void;
  onStartTraining: () => void;
  onResetModel: () => void;
  onTriggerPulse: () => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  architecture,
  modelParamsCount,
  snapshot,
  dataset,
  trainingConfig,
  trainingStatus,
  currentEpoch,
  currentLoss,
  currentAccuracy,
  logs,
  onNavigate,
  onStartTraining,
  onResetModel,
  onTriggerPulse,
}) => {
  const totalLayers = calculateTotalLayers(architecture);
  const totalNeurons = calculateTotalNeurons(architecture);
  const totalParams = modelParamsCount || calculateTotalParameters(architecture);
  const isTraining = trainingStatus === 'training';
  const recentLogs = logs.slice(-8).reverse();

  return (
    <div className="np-dashboard">
      <div className="page-header-bar">
        <div>
          <h1 className="page-title">01 // ГЛАВНАЯ ИНФОРМАЦИОННАЯ ПАНЕЛЬ</h1>
          <p className="page-subtitle">
            Оперативная сводка топологии нейросети, параметров вычислительного графа и журнала операций
          </p>
        </div>
        <div className="np-dash-quick-actions">
          <button
            type="button"
            className="retro-btn"
            onClick={() => onNavigate('network')}
          >
            КОНСТРУКТОР СЕТИ
            <ArrowRight size={14} />
          </button>
          <button
            type="button"
            className="retro-btn"
            onClick={onResetModel}
            disabled={isTraining}
            title="Переинициализировать веса модели случайными значениями"
          >
            <RotateCcw size={14} />
            СБРОС ВЕСОВ
          </button>
        </div>
      </div>

      <div className="np-stat-grid">
        <StatCard
          code="TOP.01"
          label="Слои нейросети"
          value={totalLayers}
          subtext={`Вход: 1 • Скрытых: ${architecture.hiddenLayers.length} • Выход: 1`}
          icon={<Layers size={15} />}
        />
        <StatCard
          code="TOP.02"
          label="Всего нейронов"
          value={totalNeurons}
          subtext={`Распределение: [${snapshot.layerSizes.join(' - ')}]`}
          icon={<Cpu size={15} />}
        />
        <StatCard
          code="PAR.03"
          label="Параметры модели"
          value={totalParams}
          subtext="Обучаемые веса W и смещения b"
          accent
          icon={<Sliders size={15} />}
        />
        <StatCard
          code="TRN.04"
          label="Статус обучения"
          value={getTrainingStatusLabel(trainingStatus)}
          subtext={
            currentEpoch > 0
              ? `Эпоха ${currentEpoch}/${trainingConfig.epochs} • Loss: ${formatMetric(currentLoss, 4)} • Acc: ${formatPercent(currentAccuracy, 1)}`
              : 'Модель скомпилирована и готова к запуску'
          }
          accent={isTraining}
          icon={<Activity size={15} />}
        />
      </div>

      <div className="np-dash-main-grid">
        <div className="np-dash-vis-col">
          <NetworkVisualizer
            snapshot={snapshot}
            isTraining={isTraining}
            height={420}
            onTriggerPulse={onTriggerPulse}
            title="АРХИТЕКТУРА И АКТИВАЦИИ ТЕКУЩЕЙ МОДЕЛИ"
          />
        </div>

        <div className="np-dash-side-col">
          <div className="tech-panel np-model-card">
            <div className="tech-panel-header">
              <div className="tech-panel-title">
                <span className="tech-panel-code">[СПЕЦ]</span>
                <span>ПАСПОРТ ТЕКУЩЕЙ МОДЕЛИ</span>
              </div>
              <span className="tech-badge tech-badge-active">TF.SEQUENTIAL</span>
            </div>

            <div className="np-model-card-body">
              <div className="np-spec-row">
                <span>Схема слоёв:</span>
                <strong>{formatArchitectureString(architecture)}</strong>
              </div>
              <div className="np-spec-row">
                <span>Набор данных:</span>
                <strong>
                  {dataset.meta.name} ({dataset.points.length} точек)
                </strong>
              </div>
              <div className="np-spec-row">
                <span>Тип задачи:</span>
                <strong>
                  {dataset.meta.taskType === 'classification'
                    ? 'Бинарная классификация'
                    : 'Непрерывная регрессия'}
                </strong>
              </div>
              <div className="np-spec-row">
                <span>Оптимизатор:</span>
                <strong>{getOptimizerLabel(trainingConfig.optimizer)}</strong>
              </div>
              <div className="np-spec-row">
                <span>Скорость обучения (LR):</span>
                <strong className="phosphor-text">{trainingConfig.learningRate}</strong>
              </div>
              <div className="np-spec-row">
                <span>Целевые эпохи / Батч:</span>
                <strong>
                  {trainingConfig.epochs} эпох / {trainingConfig.batchSize} сэмплов
                </strong>
              </div>
              <div className="np-spec-row">
                <span>Текущая ошибка (Loss):</span>
                <strong>{formatMetric(currentLoss, 5)}</strong>
              </div>
              <div className="np-spec-row">
                <span>Текущая точность:</span>
                <strong className="phosphor-text">{formatPercent(currentAccuracy, 1)}</strong>
              </div>

              <hr className="tech-divider" />

              <div className="np-card-btn-row">
                <button
                  type="button"
                  className="retro-btn"
                  onClick={() => onNavigate('datasets')}
                >
                  <Database size={13} />
                  ВЫБОР ДАННЫХ
                </button>
                <button
                  type="button"
                  className="retro-btn retro-btn-primary"
                  onClick={() => {
                    if (!isTraining) {
                      onStartTraining();
                    }
                    onNavigate('training');
                  }}
                >
                  <Play size={13} />
                  {isTraining ? 'К ПРОЦЕССУ' : 'ОБУЧИТЬ СЕТЬ'}
                </button>
              </div>
            </div>
          </div>

          <div className="tech-panel np-ops-card">
            <div className="tech-panel-header">
              <div className="tech-panel-title">
                <span className="tech-panel-code">[ЛОГ]</span>
                <span>ПОСЛЕДНИЕ ОПЕРАЦИИ</span>
              </div>
              <span>[{recentLogs.length}]</span>
            </div>

            <div className="np-ops-list">
              {recentLogs.length === 0 ? (
                <div className="np-ops-empty">Операции отсутствуют.</div>
              ) : (
                recentLogs.map((item) => (
                  <div
                    key={item.id}
                    className={`np-ops-item ${item.isError ? 'is-error' : ''}`}
                  >
                    <div className="np-ops-meta">
                      <span>[{item.timestamp}]</span>
                      <span className="np-ops-cat">{item.category}</span>
                    </div>
                    <div className="np-ops-msg">{item.message}</div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
