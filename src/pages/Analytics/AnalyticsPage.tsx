import React, { useState } from 'react';
import {
  Play,
  TrendingDown,
  Target,
  Clock,
  Hash,
  FileBarChart2,
  ArrowRight,
} from 'lucide-react';
import { TrainingCharts } from '../../components/Charts/TrainingCharts';
import { StatCard } from '../../components/Statistics/StatCard';
import {
  EpochMetric,
  GeneratedDataset,
  NetworkArchitecture,
  PageId,
  TrainingConfig,
  TrainingRunSummary,
  TrainingStatus,
} from '../../types';
import {
  calculateTotalParameters,
  formatArchitectureString,
  formatMetric,
  formatPercent,
  getOptimizerLabel,
} from '../../utils/formatters';
import './AnalyticsPage.css';

interface AnalyticsPageProps {
  liveHistory: EpochMetric[];
  runsHistory: TrainingRunSummary[];
  trainingStatus: TrainingStatus;
  architecture: NetworkArchitecture;
  dataset: GeneratedDataset;
  trainingConfig: TrainingConfig;
  onStartTraining: () => void;
  onNavigate: (page: PageId) => void;
}

export const AnalyticsPage: React.FC<AnalyticsPageProps> = ({
  liveHistory,
  runsHistory,
  trainingStatus,
  architecture,
  dataset,
  trainingConfig,
  onStartTraining,
  onNavigate,
}) => {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const isTraining = trainingStatus === 'training';

  const latestRun = runsHistory.length > 0 ? runsHistory[runsHistory.length - 1] : null;
  const inspectedRun =
    !isTraining && selectedRunId
      ? runsHistory.find((r) => r.id === selectedRunId) ?? latestRun
      : latestRun;

  // Во время активного обучения отображаем живой поток метрик, иначе выбранный/последний запуск
  const displayedHistory =
    isTraining || liveHistory.length > 0 && !selectedRunId
      ? liveHistory
      : inspectedRun?.history ?? liveHistory;

  const firstPoint = displayedHistory[0] ?? null;
  const lastPoint =
    displayedHistory.length > 0 ? displayedHistory[displayedHistory.length - 1] : null;

  const lossDelta =
    firstPoint && lastPoint ? firstPoint.loss - lastPoint.loss : null;
  const accDelta =
    firstPoint && lastPoint ? lastPoint.accuracy - firstPoint.accuracy : null;

  return (
    <div className="np-analytics-page">
      <div className="page-header-bar">
        <div>
          <h1 className="page-title">05 // АНАЛИТИКА И РЕЗУЛЬТАТЫ ОБУЧЕНИЯ</h1>
          <p className="page-subtitle">
            Реальные телеметрические кривые сходимости TensorFlow.js, статистика эпох и архив сессий
          </p>
        </div>

        <div className="np-an-header-btns">
          <button
            type="button"
            className="retro-btn"
            onClick={() => onNavigate('training')}
          >
            ПАРАМЕТРЫ ОБУЧЕНИЯ
            <ArrowRight size={14} />
          </button>
          {!isTraining && (
            <button
              type="button"
              className="retro-btn retro-btn-primary"
              onClick={() => {
                setSelectedRunId(null);
                onStartTraining();
              }}
            >
              <Play size={14} />
              ЗАПУСТИТЬ НОВЫЙ ЦИКЛ
            </button>
          )}
        </div>
      </div>

      {/* Верхние карточки реальных показателей */}
      <div className="np-stat-grid">
        <StatCard
          code="ANL.01"
          label="Количество эпох"
          value={lastPoint ? lastPoint.epoch : 0}
          subtext={
            isTraining
              ? `Идёт запись из ${trainingConfig.epochs} целевых`
              : inspectedRun
                ? `План: ${inspectedRun.targetEpochs} эпох`
                : 'Обучение ещё не запускалось'
          }
          icon={<Hash size={15} />}
        />
        <StatCard
          code="ANL.02"
          label="Финальная ошибка (Loss)"
          value={lastPoint ? formatMetric(lastPoint.loss, 5) : '—'}
          subtext={
            lossDelta !== null
              ? `Снижение ошибки: ${lossDelta >= 0 ? '-' : '+'}${Math.abs(lossDelta).toFixed(4)}`
              : 'Ожидание данных обучения'
          }
          icon={<TrendingDown size={15} />}
        />
        <StatCard
          code="ANL.03"
          label="Достигнутая точность"
          value={lastPoint ? formatPercent(lastPoint.accuracy, 1) : '—'}
          subtext={
            accDelta !== null
              ? `Прирост от 1-й эпохи: ${accDelta >= 0 ? '+' : ''}${(accDelta * 100).toFixed(1)}%`
              : 'Реальные вычисления TF.js'
          }
          accent={Boolean(lastPoint)}
          icon={<Target size={15} />}
        />
        <StatCard
          code="ANL.04"
          label="Длительность расчёта"
          value={lastPoint ? `${(lastPoint.elapsedMs / 1000).toFixed(2)} с` : '—'}
          subtext={
            lastPoint && lastPoint.elapsedMs > 0
              ? `Скорость: ${Math.round((lastPoint.epoch / lastPoint.elapsedMs) * 1000)} эпох/сек`
              : `Выборка: ${dataset.meta.name}`
          }
          icon={<Clock size={15} />}
        />
      </div>

      {displayedHistory.length === 0 ? (
        <div className="tech-panel np-an-empty-state">
          <div className="np-an-empty-icon">
            <FileBarChart2 size={28} />
          </div>
          <h2 className="np-an-empty-title">ДАННЫЕ ОБУЧЕНИЯ ЕЩЁ НЕ ЗАПИСАНЫ</h2>
          <p className="np-an-empty-desc">
            В соответствии с лабораторным регламентом фиктивные графики не отображаются.
            Текущая модель ({formatArchitectureString(architecture)},{' '}
            {calculateTotalParameters(architecture)} параметров) скомпилирована для выборки «
            {dataset.meta.name}» и готова к настоящему обучению в браузере.
          </p>
          <button
            type="button"
            className="retro-btn retro-btn-primary"
            onClick={onStartTraining}
          >
            <Play size={14} />
            ЗАПУСТИТЬ ОБУЧЕНИЕ И ПОСТРОИТЬ ГРАФИКИ
          </button>
        </div>
      ) : (
        <>
          <TrainingCharts
            history={displayedHistory}
            accuracyLabel={
              (inspectedRun?.taskType ?? dataset.meta.taskType) === 'classification'
                ? 'ИЗМЕНЕНИЕ ТОЧНОСТИ КЛАССИФИКАЦИИ (%)'
                : 'ИЗМЕНЕНИЕ ТОЧНОСТИ АППРОКСИМАЦИИ R² (%)'
            }
          />

          <div className="np-an-bottom-grid">
            {/* Сводный отчёт последнего / выбранного обучения */}
            <div className="tech-panel">
              <div className="tech-panel-header">
                <div className="tech-panel-title">
                  <span className="tech-panel-code">[ОТЧЁТ]</span>
                  <span>
                    {inspectedRun
                      ? `РЕЗУЛЬТАТЫ ЗАПУСКА #${inspectedRun.runNumber}`
                      : 'ТЕКУЩАЯ СЕССИЯ ОБУЧЕНИЯ'}
                  </span>
                </div>
                <span className="tech-badge tech-badge-active">
                  {isTraining
                    ? 'В ПРОЦЕССЕ'
                    : inspectedRun?.status === 'stopped'
                      ? 'ОСТАНОВЛЕНО'
                      : 'ЗАВЕРШЕНО'}
                </span>
              </div>

              <div className="np-an-report-body">
                <div className="np-spec-row">
                  <span>Набор данных:</span>
                  <strong>{inspectedRun?.datasetName ?? dataset.meta.name}</strong>
                </div>
                <div className="np-spec-row">
                  <span>Топология сети:</span>
                  <strong>
                    {inspectedRun?.architectureSummary ??
                      formatArchitectureString(architecture)}
                  </strong>
                </div>
                <div className="np-spec-row">
                  <span>Количество параметров:</span>
                  <strong>
                    {inspectedRun?.totalParams ?? calculateTotalParameters(architecture)}
                  </strong>
                </div>
                <div className="np-spec-row">
                  <span>Оптимизатор и LR:</span>
                  <strong>
                    {getOptimizerLabel(inspectedRun?.optimizer ?? trainingConfig.optimizer)} •{' '}
                    LR {inspectedRun?.learningRate ?? trainingConfig.learningRate}
                  </strong>
                </div>
                <div className="np-spec-row">
                  <span>Начальная → Финальная ошибка:</span>
                  <strong>
                    {formatMetric(firstPoint?.loss, 4)} →{' '}
                    <span className="phosphor-text">{formatMetric(lastPoint?.loss, 5)}</span>
                  </strong>
                </div>
                <div className="np-spec-row">
                  <span>Начальная → Финальная точность:</span>
                  <strong>
                    {formatPercent(firstPoint?.accuracy, 1)} →{' '}
                    <span className="phosphor-text">
                      {formatPercent(lastPoint?.accuracy, 1)}
                    </span>
                  </strong>
                </div>
              </div>
            </div>

            {/* Таблица всех запусков в текущей сессии */}
            <div className="tech-panel">
              <div className="tech-panel-header">
                <div className="tech-panel-title">
                  <span className="tech-panel-code">[АРХИВ]</span>
                  <span>ЖУРНАЛ ЭКСПЕРИМЕНТОВ СЕССИИ</span>
                </div>
                <span>[{runsHistory.length}]</span>
              </div>

              {runsHistory.length === 0 ? (
                <div className="np-an-no-runs">
                  Идёт первый цикл обучения. Завершённые прогоны появятся здесь.
                </div>
              ) : (
                <div className="np-param-table-wrap">
                  <table className="np-tech-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Время</th>
                        <th>Выборка</th>
                        <th>Эпохи</th>
                        <th>Loss</th>
                        <th>Точность</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...runsHistory].reverse().map((run) => {
                        const isSel =
                          (selectedRunId && run.id === selectedRunId) ||
                          (!selectedRunId && latestRun?.id === run.id);
                        return (
                          <tr
                            key={run.id}
                            className={`np-run-row ${isSel ? 'is-selected' : ''}`}
                            onClick={() => setSelectedRunId(run.id)}
                            title="Нажмите для просмотра графиков этого запуска"
                          >
                            <td className="phosphor-text">#{run.runNumber}</td>
                            <td>{run.startedAt}</td>
                            <td>{run.datasetName}</td>
                            <td>
                              {run.epochsCompleted}/{run.targetEpochs}
                            </td>
                            <td>{run.finalLoss.toFixed(4)}</td>
                            <td className="phosphor-text">
                              {(run.finalAccuracy * 100).toFixed(1)}%
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
};
