import React from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import { EpochMetric } from '../../types';
import './Charts.css';

interface TrainingChartsProps {
  history: EpochMetric[];
  accuracyLabel?: string;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: EpochMetric }>;
  mode: 'loss' | 'accuracy';
}

const RetroTooltip: React.FC<CustomTooltipProps> = ({ active, payload, mode }) => {
  if (!active || !payload || payload.length === 0) return null;
  const pt = payload[0].payload;
  return (
    <div className="np-recharts-tooltip">
      <div className="np-recharts-tooltip-title">ЭПОХА #{pt.epoch}</div>
      {mode === 'loss' ? (
        <div className="np-recharts-tooltip-row">
          <span>Ошибка (Loss):</span>
          <strong className="phosphor-text">{pt.loss.toFixed(5)}</strong>
        </div>
      ) : (
        <div className="np-recharts-tooltip-row">
          <span>Точность:</span>
          <strong className="phosphor-text">{(pt.accuracy * 100).toFixed(2)}%</strong>
        </div>
      )}
      <div className="np-recharts-tooltip-row">
        <span>Время от старта:</span>
        <span>{(pt.elapsedMs / 1000).toFixed(2)} с</span>
      </div>
    </div>
  );
};

export const TrainingCharts: React.FC<TrainingChartsProps> = ({
  history,
  accuracyLabel = 'ДИНАМИКА ТОЧНОСТИ МОДЕЛИ (%)',
}) => {
  const chartData = history.map((item) => ({
    ...item,
    accuracyPct: Number((item.accuracy * 100).toFixed(2)),
  }));

  return (
    <div className="np-analytics-charts-grid">
      {/* График ошибки Loss */}
      <div className="tech-panel">
        <div className="tech-panel-header">
          <div className="tech-panel-title">
            <span className="tech-panel-code">[ГРАФИК 01]</span>
            <span>ИЗМЕНЕНИЕ ОШИБКИ ОБУЧЕНИЯ (LOSS)</span>
          </div>
          {history.length > 0 && (
            <span className="phosphor-text">
              МИН: {Math.min(...history.map((h) => h.loss)).toFixed(5)}
            </span>
          )}
        </div>

        <div className="np-chart-box" style={{ height: 295 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 16, left: 4, bottom: 6 }}>
              <defs>
                <linearGradient id="lossPhosphorFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#39FF78" stopOpacity={0.28} />
                  <stop offset="95%" stopColor="#39FF78" stopOpacity={0.01} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#1E3F28" strokeDasharray="3 3" />
              <XAxis
                dataKey="epoch"
                stroke="#718575"
                tick={{ fill: '#718575', fontSize: 11, fontFamily: 'JetBrains Mono' }}
              />
              <YAxis
                stroke="#718575"
                tick={{ fill: '#718575', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                domain={['auto', 'auto']}
                width={54}
              />
              <Tooltip content={<RetroTooltip mode="loss" />} />
              <Area
                type="monotone"
                dataKey="loss"
                stroke="#39FF78"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#lossPhosphorFill)"
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* График точности Accuracy */}
      <div className="tech-panel">
        <div className="tech-panel-header">
          <div className="tech-panel-title">
            <span className="tech-panel-code">[ГРАФИК 02]</span>
            <span>{accuracyLabel}</span>
          </div>
          {history.length > 0 && (
            <span className="phosphor-text">
              МАКС: {(Math.max(...history.map((h) => h.accuracy)) * 100).toFixed(1)}%
            </span>
          )}
        </div>

        <div className="np-chart-box" style={{ height: 295 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 16, left: 4, bottom: 6 }}>
              <defs>
                <linearGradient id="accPhosphorFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8CBF98" stopOpacity={0.28} />
                  <stop offset="95%" stopColor="#8CBF98" stopOpacity={0.01} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#1E3F28" strokeDasharray="3 3" />
              <XAxis
                dataKey="epoch"
                stroke="#718575"
                tick={{ fill: '#718575', fontSize: 11, fontFamily: 'JetBrains Mono' }}
              />
              <YAxis
                stroke="#718575"
                tick={{ fill: '#718575', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                domain={[0, 100]}
                unit="%"
                width={54}
              />
              <Tooltip content={<RetroTooltip mode="accuracy" />} />
              <Area
                type="monotone"
                dataKey="accuracyPct"
                stroke="#39FF78"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#accPhosphorFill)"
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
