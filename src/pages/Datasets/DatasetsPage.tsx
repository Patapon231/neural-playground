import React from 'react';
import { RefreshCw, ArrowRight, Database, CheckCircle2, Info } from 'lucide-react';
import { DatasetPlot } from '../../components/Charts/DatasetPlot';
import { DATASET_LIST } from '../../services/datasets';
import {
  DatasetConfig,
  DatasetId,
  GeneratedDataset,
  PageId,
  PredictionVisualData,
} from '../../types';
import './DatasetsPage.css';

interface DatasetsPageProps {
  dataset: GeneratedDataset;
  predictionVisuals: PredictionVisualData | null;
  isTraining: boolean;
  onSelectDataset: (id: DatasetId) => void;
  onUpdateDatasetConfig: (patch: Partial<DatasetConfig>) => void;
  onRegenerate: () => void;
  onProbeCoordinate: (inputVec: number[]) => void;
  onNavigate: (page: PageId) => void;
}

export const DatasetsPage: React.FC<DatasetsPageProps> = ({
  dataset,
  predictionVisuals,
  isTraining,
  onSelectDataset,
  onUpdateDatasetConfig,
  onRegenerate,
  onProbeCoordinate,
  onNavigate,
}) => {
  const { meta, config, points } = dataset;

  const classOneCount =
    meta.taskType === 'classification'
      ? points.filter((p) => p.label >= 0.5).length
      : points.length;
  const classZeroCount =
    meta.taskType === 'classification' ? points.length - classOneCount : 0;

  return (
    <div className="np-datasets-page">
      <div className="page-header-bar">
        <div>
          <h1 className="page-title">03 // БИБЛИОТЕКА НАБОРОВ ДАННЫХ</h1>
          <p className="page-subtitle">
            Синтетические лабораторные выборки для проверки линейной и нелинейной разделимости моделей
          </p>
        </div>

        <div className="np-ds-header-actions">
          <button
            type="button"
            className="retro-btn"
            onClick={onRegenerate}
            disabled={isTraining}
          >
            <RefreshCw size={14} />
            ПЕРЕГЕНЕРИРОВАТЬ ТОЧКИ
          </button>
          <button
            type="button"
            className="retro-btn retro-btn-primary"
            onClick={() => onNavigate('training')}
          >
            ПЕРЕЙТИ К ОБУЧЕНИЮ
            <ArrowRight size={14} />
          </button>
        </div>
      </div>

      {/* Карточки выбора набора данных */}
      <div className="np-ds-cards-grid">
        {DATASET_LIST.map((item) => {
          const selected = item.id === meta.id;
          return (
            <button
              key={item.id}
              type="button"
              className={`tech-panel np-ds-card ${selected ? 'is-selected' : ''}`}
              disabled={isTraining}
              onClick={() => onSelectDataset(item.id)}
            >
              <div className="np-ds-card-top">
                <span className="tech-badge tech-badge-active">{item.code}</span>
                <span className="np-ds-diff">СЛОЖНОСТЬ: {item.difficulty}</span>
              </div>
              <div className="np-ds-card-title">
                <Database size={14} />
                <span>{item.name}</span>
                {selected && <CheckCircle2 size={14} className="np-ds-check" />}
              </div>
              <p className="np-ds-card-desc">{item.description}</p>
              <div className="np-ds-card-foot">
                <span>
                  Вход: {item.inputDim === 2 ? '2 признака (X₁, X₂)' : '1 признак (X)'}
                </span>
                <span>
                  {item.taskType === 'classification' ? 'Классификация' : 'Регрессия'}
                </span>
              </div>
            </button>
          );
        })}
      </div>

      <div className="np-ds-main-grid">
        <div className="np-ds-plot-col">
          <DatasetPlot
            meta={meta}
            points={points}
            predictionVisuals={predictionVisuals}
            height={420}
            onProbeCoordinate={onProbeCoordinate}
          />
        </div>

        <div className="np-ds-controls-col">
          <div className="tech-panel np-ds-params-panel">
            <div className="tech-panel-header">
              <div className="tech-panel-title">
                <span className="tech-panel-code">[ГЕНЕРАТОР]</span>
                <span>ПАРАМЕТРЫ ВЫБОРКИ</span>
              </div>
            </div>

            <div className="np-ds-params-body">
              <div className="np-ds-control">
                <div className="np-ds-control-head">
                  <span>Объём выборки (количество точек):</span>
                  <strong className="phosphor-text">{config.sampleCount}</strong>
                </div>
                <input
                  type="range"
                  className="retro-range"
                  min={80}
                  max={400}
                  step={20}
                  value={config.sampleCount}
                  disabled={isTraining}
                  onChange={(e) =>
                    onUpdateDatasetConfig({ sampleCount: Number(e.target.value) })
                  }
                />
                <div className="np-ds-scale">
                  <span>80</span>
                  <span>200</span>
                  <span>400</span>
                </div>
              </div>

              <div className="np-ds-control">
                <div className="np-ds-control-head">
                  <span>Уровень гауссова шума:</span>
                  <strong className="phosphor-text">
                    {Math.round(config.noise * 100)}%
                  </strong>
                </div>
                <input
                  type="range"
                  className="retro-range"
                  min={0}
                  max={0.3}
                  step={0.02}
                  value={config.noise}
                  disabled={isTraining}
                  onChange={(e) =>
                    onUpdateDatasetConfig({ noise: Number(e.target.value) })
                  }
                />
                <div className="np-ds-scale">
                  <span>0% (эталон)</span>
                  <span>15%</span>
                  <span>30% (высокий шум)</span>
                </div>
              </div>

              <hr className="tech-divider" />

              <div className="np-ds-stats-box">
                <div className="np-spec-row">
                  <span>Активный набор:</span>
                  <strong>{meta.name}</strong>
                </div>
                <div className="np-spec-row">
                  <span>Всего наблюдений:</span>
                  <strong>{points.length}</strong>
                </div>
                {meta.taskType === 'classification' ? (
                  <>
                    <div className="np-spec-row">
                      <span>Класс 1 (положительный):</span>
                      <strong className="phosphor-text">{classOneCount} точек</strong>
                    </div>
                    <div className="np-spec-row">
                      <span>Класс 0 (фоновый):</span>
                      <strong>{classZeroCount} точек</strong>
                    </div>
                  </>
                ) : (
                  <div className="np-spec-row">
                    <span>Целевая формула:</span>
                    <strong className="phosphor-text">Y ≈ 0.68·X + 0.12 + ε</strong>
                  </div>
                )}
                <div className="np-spec-row">
                  <span>Диапазон координат:</span>
                  <strong>[-1.05 .. +1.05]</strong>
                </div>
              </div>
            </div>
          </div>

          <div className="tech-panel np-ds-hint-panel">
            <div className="tech-panel-header">
              <div className="tech-panel-title">
                <Info size={13} />
                <span>МЕТОДИЧЕСКАЯ СПРАВКА</span>
              </div>
            </div>
            <div className="np-ds-hint-body">
              <p>{meta.recommendedHint}</p>
              <p className="np-ds-probe-tip">
                Совет: щёлкните левой кнопкой мыши в любой точке 2D-графика, чтобы подать
                координаты на вход нейросети и проверить отклик нейронов.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
