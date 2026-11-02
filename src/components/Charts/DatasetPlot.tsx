import React, { useEffect, useRef, useState } from 'react';
import { Crosshair } from 'lucide-react';
import { DataPoint, DatasetMetadata, PredictionVisualData } from '../../types';
import './Charts.css';

interface DatasetPlotProps {
  meta: DatasetMetadata;
  points: DataPoint[];
  predictionVisuals: PredictionVisualData | null;
  height?: number;
  onProbeCoordinate?: (inputVec: number[]) => void;
  title?: string;
}

const DOMAIN_MIN = -1.1;
const DOMAIN_MAX = 1.1;

export const DatasetPlot: React.FC<DatasetPlotProps> = ({
  meta,
  points,
  predictionVisuals,
  height = 380,
  onProbeCoordinate,
  title,
}) => {
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [size, setSize] = useState<{ width: number; height: number }>({
    width: 460,
    height,
  });
  const [cursorCoords, setCursorCoords] = useState<{ x: number; y: number } | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const sync = () => {
      const rect = el.getBoundingClientRect();
      if (rect.width > 0) {
        setSize({ width: Math.floor(rect.width), height });
      }
    };
    sync();
    const obs = new ResizeObserver(sync);
    obs.observe(el);
    return () => obs.disconnect();
  }, [height]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = size.width;
    const h = size.height;

    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    ctx.save();
    ctx.scale(dpr, dpr);

    ctx.fillStyle = '#080B08';
    ctx.fillRect(0, 0, w, h);

    const padL = 42;
    const padR = 18;
    const padT = 18;
    const padB = 34;
    const plotW = Math.max(80, w - padL - padR);
    const plotH = Math.max(80, h - padT - padB);

    const toPxX = (val: number) =>
      padL + ((val - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)) * plotW;
    const toPxY = (val: number) =>
      padT + (1 - (val - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN)) * plotH;

    // 1. Карта решений для 2D классификации
    if (
      meta.taskType === 'classification' &&
      predictionVisuals?.gridSize &&
      predictionVisuals.gridValues &&
      predictionVisuals.gridValues.length ===
        predictionVisuals.gridSize * predictionVisuals.gridSize
    ) {
      const g = predictionVisuals.gridSize;
      const cellW = plotW / g;
      const cellH = plotH / g;
      const vals = predictionVisuals.gridValues;

      for (let r = 0; r < g; r++) {
        for (let c = 0; c < g; c++) {
          const prob = vals[r * g + c];
          const cx = padL + c * cellW;
          const cy = padT + r * cellH;
          if (prob >= 0.5) {
            const strength = (prob - 0.5) * 2;
            ctx.fillStyle = `rgba(57, 255, 120, ${(0.04 + strength * 0.22).toFixed(3)})`;
          } else {
            const strength = (0.5 - prob) * 2;
            ctx.fillStyle = `rgba(18, 34, 22, ${(0.12 + strength * 0.35).toFixed(3)})`;
          }
          ctx.fillRect(cx, cy, Math.ceil(cellW), Math.ceil(cellH));

          // Подсветка границы решений P ≈ 0.5
          if (Math.abs(prob - 0.5) < 0.045) {
            ctx.fillStyle = 'rgba(140, 191, 152, 0.26)';
            ctx.fillRect(cx, cy, Math.ceil(cellW), Math.ceil(cellH));
          }
        }
      }
    }

    // 2. Координатная сетка и оси
    ctx.strokeStyle = 'rgba(40, 91, 55, 0.35)';
    ctx.lineWidth = 1;
    const ticks = [-1, -0.5, 0, 0.5, 1];

    ticks.forEach((t) => {
      const px = toPxX(t);
      const py = toPxY(t);

      ctx.beginPath();
      ctx.setLineDash(t === 0 ? [] : [3, 4]);
      ctx.strokeStyle = t === 0 ? 'rgba(57, 255, 120, 0.35)' : 'rgba(40, 91, 55, 0.32)';
      ctx.moveTo(px, padT);
      ctx.lineTo(px, padT + plotH);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(padL, py);
      ctx.lineTo(padL + plotW, py);
      ctx.stroke();
      ctx.setLineDash([]);

      // Подписи осей
      ctx.fillStyle = '#718575';
      ctx.font = '500 10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(t.toFixed(1), px, padT + plotH + 7);

      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(t.toFixed(1), padL - 7, py);
    });

    // Рамка области построения
    ctx.strokeStyle = '#285B37';
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, plotW, plotH);

    // 3. Отрисовка кривой регрессии (если линейная зависимость)
    if (
      meta.taskType === 'regression' &&
      predictionVisuals?.regressionCurve &&
      predictionVisuals.regressionCurve.length > 1
    ) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(padL, padT, plotW, plotH);
      ctx.clip();

      ctx.beginPath();
      predictionVisuals.regressionCurve.forEach((pt, idx) => {
        const px = toPxX(pt.x);
        const py = toPxY(pt.y);
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.strokeStyle = '#39FF78';
      ctx.lineWidth = 2.4;
      ctx.shadowColor = 'rgba(57, 255, 120, 0.55)';
      ctx.shadowBlur = 7;
      ctx.stroke();
      ctx.restore();
    }

    // 4. Отрисовка точек выборки
    points.forEach((pt) => {
      const px = toPxX(pt.x);
      const py = toPxY(pt.y);

      if (meta.taskType === 'classification') {
        if (pt.label >= 0.5) {
          // Класс 1: яркий фосфорный узел
          ctx.beginPath();
          ctx.arc(px, py, 4.2, 0, Math.PI * 2);
          ctx.fillStyle = '#39FF78';
          ctx.fill();
          ctx.strokeStyle = '#080B08';
          ctx.lineWidth = 1;
          ctx.stroke();
        } else {
          // Класс 0: полый технический квадрат
          const s = 7;
          ctx.fillStyle = '#080B08';
          ctx.fillRect(px - s / 2, py - s / 2, s, s);
          ctx.strokeStyle = '#8CBF98';
          ctx.lineWidth = 1.5;
          ctx.strokeRect(px - s / 2, py - s / 2, s, s);
        }
      } else {
        // Точки регрессии
        ctx.beginPath();
        ctx.arc(px, py, 3.4, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(140, 191, 152, 0.85)';
        ctx.fill();
        ctx.strokeStyle = '#080B08';
        ctx.lineWidth = 0.9;
        ctx.stroke();
      }
    });

    // Подписи осей X и Y
    ctx.fillStyle = '#8CBF98';
    ctx.font = '700 10px "IBM Plex Mono", monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'bottom';
    ctx.fillText(meta.inputDim === 2 ? 'ОСЬ X₁' : 'ПРИЗНАК X', padL + plotW - 4, padT + plotH - 4);

    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(meta.inputDim === 2 ? 'ОСЬ X₂' : 'ЦЕЛЬ Y', padL + 6, padT + 5);

    ctx.restore();
  }, [meta, points, predictionVisuals, size]);

  const getLogicalCoords = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const padL = 42;
    const padR = 18;
    const padT = 18;
    const padB = 34;
    const plotW = Math.max(80, size.width - padL - padR);
    const plotH = Math.max(80, size.height - padT - padB);

    if (mx < padL || mx > padL + plotW || my < padT || my > padT + plotH) {
      return null;
    }

    const x = DOMAIN_MIN + ((mx - padL) / plotW) * (DOMAIN_MAX - DOMAIN_MIN);
    const y = DOMAIN_MAX - ((my - padT) / plotH) * (DOMAIN_MAX - DOMAIN_MIN);
    return { x, y };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setCursorCoords(getLogicalCoords(e));
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onProbeCoordinate) return;
    const coords = getLogicalCoords(e);
    if (!coords) return;
    if (meta.inputDim === 2) {
      onProbeCoordinate([coords.x, coords.y]);
    } else {
      onProbeCoordinate([coords.x]);
    }
  };

  return (
    <div className="tech-panel np-plot-panel">
      <div className="tech-panel-header">
        <div className="tech-panel-title">
          <span className="tech-panel-code">[ПЛОСКОСТЬ 2D]</span>
          <span>{title || `ПРОСТРАНСТВО ПРИЗНАКОВ: ${meta.name.toUpperCase()}`}</span>
        </div>
        <div className="np-plot-coords">
          <Crosshair size={13} />
          {cursorCoords ? (
            <span>
              X={cursorCoords.x.toFixed(2)}, Y={cursorCoords.y.toFixed(2)}
            </span>
          ) : (
            <span>Кликните для зондирования сети</span>
          )}
        </div>
      </div>

      <div className="np-plot-canvas-wrap" ref={wrapRef} style={{ height }}>
        <canvas
          ref={canvasRef}
          className="np-plot-canvas"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setCursorCoords(null)}
          onClick={handleClick}
        />
      </div>

      <div className="np-plot-legend">
        {meta.taskType === 'classification' ? (
          <>
            <span className="np-plot-leg-item">
              <span className="np-marker-dot" /> Класс 1 (активный сигнал)
            </span>
            <span className="np-plot-leg-item">
              <span className="np-marker-sq" /> Класс 0 (фоновый класс)
            </span>
            <span className="np-plot-leg-item">
              Фон = разделяющая поверхность модели P(Ŷ)
            </span>
          </>
        ) : (
          <>
            <span className="np-plot-leg-item">
              <span className="np-marker-reg" /> Наблюдения выборки (X, Y)
            </span>
            <span className="np-plot-leg-item">
              <span className="np-marker-line" /> Аппроксимация нейросети Ŷ(X)
            </span>
          </>
        )}
      </div>
    </div>
  );
};
