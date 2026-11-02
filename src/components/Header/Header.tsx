import React from 'react';
import { Play, Square, Monitor, Cpu, Activity } from 'lucide-react';
import { TrainingStatus } from '../../types';
import { formatMetric, formatPercent, getTrainingStatusLabel } from '../../utils/formatters';
import './Header.css';

interface HeaderProps {
  systemReady: boolean;
  tfBackend: string;
  totalLayers: number;
  totalParams: number;
  trainingStatus: TrainingStatus;
  currentEpoch: number;
  targetEpochs: number;
  currentLoss: number | null;
  currentAccuracy: number | null;
  crtEnabled: boolean;
  onToggleCrt: () => void;
  onStartTraining: () => void;
  onStopTraining: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  systemReady,
  tfBackend,
  totalLayers,
  totalParams,
  trainingStatus,
  currentEpoch,
  targetEpochs,
  currentLoss,
  currentAccuracy,
  crtEnabled,
  onToggleCrt,
  onStartTraining,
  onStopTraining,
}) => {
  const isTraining = trainingStatus === 'training';

  return (
    <header className="np-header">
      <div className="np-header-brand">
        <div className="np-brand-mark" aria-hidden="true">
          <span>NP</span>
          <span className="np-brand-sub">//88</span>
        </div>
        <div className="np-brand-titles">
          <div className="np-brand-name">
            NEURAL PLAYGROUND
            <span className="blinking-cursor" aria-hidden="true" />
          </div>
          <div className="np-brand-caption">
            ИНТЕРАКТИВНАЯ ЛАБОРАТОРИЯ НЕЙРОННЫХ СЕТЕЙ
          </div>
        </div>
      </div>

      <div className="np-header-telemetry">
        <div className="np-status-block">
          <span className="np-status-label">
            <Cpu size={13} />
            СТАТУС СИСТЕМЫ
          </span>
          <span className={`np-status-value ${systemReady ? 'is-online' : 'is-warn'}`}>
            <span className="np-led-dot" />
            {systemReady ? `АКТИВНА [${tfBackend.toUpperCase()}]` : 'ИНИЦИАЛИЗАЦИЯ...'}
          </span>
        </div>

        <div className="np-header-sep" />

        <div className="np-status-block">
          <span className="np-status-label">
            <Activity size={13} />
            ТЕКУЩАЯ МОДЕЛЬ
          </span>
          <span className={`np-status-value ${isTraining ? 'is-training' : ''}`}>
            {isTraining ? (
              <>
                ЭПОХА {currentEpoch}/{targetEpochs} • LOSS {formatMetric(currentLoss, 4)} • ACC{' '}
                {formatPercent(currentAccuracy, 1)}
              </>
            ) : (
              <>
                {getTrainingStatusLabel(trainingStatus)} • {totalLayers} СЛОЯ • {totalParams} ПАРАМ.
              </>
            )}
          </span>
        </div>
      </div>

      <div className="np-header-actions">
        <button
          type="button"
          className="retro-btn retro-btn-sm"
          onClick={onToggleCrt}
          title="Переключить визуальный фильтр ЭЛТ-монитора"
        >
          <Monitor size={14} />
          ЭЛТ: {crtEnabled ? 'ВКЛ' : 'ВЫКЛ'}
        </button>

        {isTraining ? (
          <button
            type="button"
            className="retro-btn retro-btn-danger"
            onClick={onStopTraining}
          >
            <Square size={14} />
            ОСТАНОВИТЬ ОБУЧЕНИЕ
          </button>
        ) : (
          <button
            type="button"
            className="retro-btn retro-btn-primary"
            onClick={onStartTraining}
            disabled={!systemReady}
          >
            <Play size={14} />
            ЗАПУСТИТЬ ОБУЧЕНИЕ
          </button>
        )}
      </div>
    </header>
  );
};
