import React, { useEffect, useRef, useState } from 'react';
import { Zap, PlayCircle, Info } from 'lucide-react';
import { NetworkSnapshot } from '../../types';
import './NetworkVisualizer.css';

interface NetworkVisualizerProps {
  snapshot: NetworkSnapshot;
  isTraining: boolean;
  height?: number;
  onTriggerPulse?: () => void;
  title?: string;
}

interface HoveredNode {
  layerIdx: number;
  neuronIdx: number;
  x: number;
  y: number;
}

function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

export const NetworkVisualizer: React.FC<NetworkVisualizerProps> = ({
  snapshot,
  isTraining,
  height = 410,
  onTriggerPulse,
  title = 'ТОПОЛОГИЯ И ПОТОК СИГНАЛОВ НЕЙРОСЕТИ',
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [canvasSize, setCanvasSize] = useState<{ width: number; height: number }>({
    width: 760,
    height,
  });
  const [signalEnabled, setSignalEnabled] = useState<boolean>(true);
  const [hovered, setHovered] = useState<HoveredNode | null>(null);
  const nodePositionsRef = useRef<Array<Array<{ x: number; y: number; r: number }>>>([]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const updateSize = () => {
      const rect = el.getBoundingClientRect();
      if (rect.width > 0) {
        setCanvasSize({
          width: Math.floor(rect.width),
          height,
        });
      }
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(el);
    return () => observer.disconnect();
  }, [height]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animFrameId = 0;
    let phase = 0;
    let lastTime = performance.now();

    const render = (now: number) => {
      const dt = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;

      const numLayers = snapshot.layerSizes.length;
      const numTransitions = Math.max(1, numLayers - 1);

      if (signalEnabled) {
        const speed = isTraining ? 2.15 : 0.78;
        phase = (phase + dt * speed) % (numTransitions + 0.5);
      }

      const dpr = window.devicePixelRatio || 1;
      const w = canvasSize.width;
      const h = canvasSize.height;

      if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
        canvas.width = Math.floor(w * dpr);
        canvas.height = Math.floor(h * dpr);
      }

      ctx.save();
      ctx.scale(dpr, dpr);

      // 1. Очистка и координатная сетка осциллографа
      ctx.fillStyle = '#080B08';
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = 'rgba(40, 91, 55, 0.16)';
      ctx.lineWidth = 1;
      const gridStep = 28;
      for (let gx = 0; gx < w; gx += gridStep) {
        ctx.beginPath();
        ctx.moveTo(gx + 0.5, 0);
        ctx.lineTo(gx + 0.5, h);
        ctx.stroke();
      }
      for (let gy = 0; gy < h; gy += gridStep) {
        ctx.beginPath();
        ctx.moveTo(0, gy + 0.5);
        ctx.lineTo(w, gy + 0.5);
        ctx.stroke();
      }

      // Верхняя шкала направления распространения данных
      ctx.fillStyle = '#718575';
      ctx.font = '600 10px "IBM Plex Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillText('ПОТОК ДАННЫХ: ПРЯМОЕ РАСПРОСТРАНЕНИЕ (X ──► Ŷ)', 14, 19);

      if (isTraining) {
        ctx.fillStyle = '#39FF78';
        ctx.textAlign = 'right';
        ctx.fillText('● ОБНОВЛЕНИЕ ВЕСОВ АКТИВНО', w - 14, 19);
      }

      // Расчёт координат узлов
      const padLeft = Math.max(58, Math.min(86, w * 0.09));
      const padRight = Math.max(58, Math.min(86, w * 0.09));
      const padTop = 44;
      const padBottom = 56;
      const usableW = Math.max(120, w - padLeft - padRight);
      const usableH = Math.max(120, h - padTop - padBottom);

      const maxNeuronsInAnyLayer = Math.max(...snapshot.layerSizes, 1);
      const nodeRadius = clamp(
        Math.floor(usableH / (maxNeuronsInAnyLayer * 2.75)),
        10,
        21
      );

      const positions: Array<Array<{ x: number; y: number; r: number }>> = [];
      for (let l = 0; l < numLayers; l++) {
        const count = snapshot.layerSizes[l];
        const x =
          numLayers === 1
            ? w / 2
            : padLeft + (l / (numLayers - 1)) * usableW;

        const layerNodes: Array<{ x: number; y: number; r: number }> = [];
        const totalSpan = Math.min(usableH, Math.max(0, (count - 1) * (nodeRadius * 2.85)));
        const startY = padTop + usableH / 2 - totalSpan / 2;

        for (let n = 0; n < count; n++) {
          const y = count === 1 ? padTop + usableH / 2 : startY + (n / (count - 1)) * totalSpan;
          layerNodes.push({ x, y, r: nodeRadius });
        }
        positions.push(layerNodes);
      }
      nodePositionsRef.current = positions;

      // 2. Отрисовка связей и весов
      for (let l = 0; l < numLayers - 1; l++) {
        const fromLayer = positions[l];
        const toLayer = positions[l + 1];
        const layerWeights = snapshot.weights[l] || [];

        for (let i = 0; i < fromLayer.length; i++) {
          for (let j = 0; j < toLayer.length; j++) {
            const p1 = fromLayer[i];
            const p2 = toLayer[j];
            const weight = layerWeights[i]?.[j] ?? 0;
            const absW = Math.abs(weight);

            const isConnectedToHover =
              hovered !== null &&
              ((hovered.layerIdx === l && hovered.neuronIdx === i) ||
                (hovered.layerIdx === l + 1 && hovered.neuronIdx === j));

            const dimmedByHover = hovered !== null && !isConnectedToHover;

            const lineWidth = clamp(0.7 + absW * 1.65, 0.7, 4.4);
            ctx.beginPath();
            ctx.moveTo(p1.x + p1.r, p1.y);
            ctx.lineTo(p2.x - p2.r, p2.y);

            if (weight >= 0) {
              // Положительный вес: оттенки яркого зелёного люминофора
              const alpha = dimmedByHover
                ? 0.08
                : isConnectedToHover
                  ? 0.95
                  : clamp(0.2 + absW * 0.45, 0.2, 0.85);
              ctx.strokeStyle = `rgba(57, 255, 120, ${alpha.toFixed(3)})`;
              ctx.setLineDash([]);
            } else {
              // Отрицательный вес: более тёмные оттенки зелёного + пунктирный контур
              const alpha = dimmedByHover
                ? 0.08
                : isConnectedToHover
                  ? 0.95
                  : clamp(0.32 + absW * 0.42, 0.32, 0.88);
              ctx.strokeStyle = isConnectedToHover
                ? `rgba(110, 158, 122, ${alpha.toFixed(3)})`
                : `rgba(38, 86, 52, ${alpha.toFixed(3)})`;
              ctx.setLineDash([4, 3]);
            }

            ctx.lineWidth = isConnectedToHover ? lineWidth + 0.9 : lineWidth;
            ctx.stroke();
            ctx.setLineDash([]);

            // Стрелка направления сигнала на активных/выделенных связях
            if (isConnectedToHover || (fromLayer.length * toLayer.length <= 24 && !dimmedByHover)) {
              const arrowT = 0.64;
              const ax = p1.x + p1.r + (p2.x - p2.r - (p1.x + p1.r)) * arrowT;
              const ay = p1.y + (p2.y - p1.y) * arrowT;
              const angle = Math.atan2(p2.y - p1.y, p2.x - p2.r - (p1.x + p1.r));
              const arrowLen = 4.5;
              ctx.fillStyle =
                weight >= 0 ? 'rgba(57, 255, 120, 0.65)' : 'rgba(95, 145, 108, 0.55)';
              ctx.beginPath();
              ctx.moveTo(ax, ay);
              ctx.lineTo(
                ax - arrowLen * Math.cos(angle - Math.PI / 6),
                ay - arrowLen * Math.sin(angle - Math.PI / 6)
              );
              ctx.lineTo(
                ax - arrowLen * Math.cos(angle + Math.PI / 6),
                ay - arrowLen * Math.sin(angle + Math.PI / 6)
              );
              ctx.closePath();
              ctx.fill();
            }

            // Анимация распространения импульса по связи
            if (signalEnabled && !dimmedByHover) {
              const localT = phase - l;
              if (localT >= 0 && localT <= 1) {
                const sx = p1.x + p1.r + (p2.x - p2.r - (p1.x + p1.r)) * localT;
                const sy = p1.y + (p2.y - p1.y) * localT;
                const srcAct = Math.abs(snapshot.activations[l]?.[i] ?? 0.5);
                const strength = clamp(0.35 + absW * srcAct * 0.8, 0.3, 1);

                ctx.beginPath();
                ctx.arc(sx, sy, 2.2 + strength * 1.6, 0, Math.PI * 2);
                ctx.fillStyle =
                  weight >= 0
                    ? `rgba(57, 255, 120, ${(0.85 * strength).toFixed(2)})`
                    : `rgba(140, 191, 152, ${(0.6 * strength).toFixed(2)})`;
                ctx.fill();
              }
            }

            // Подпись веса при наведении на узел
            if (isConnectedToHover) {
              const mx = (p1.x + p2.x) / 2;
              const my = (p1.y + p2.y) / 2;
              const txt = (weight >= 0 ? '+' : '') + weight.toFixed(2);
              ctx.font = '600 10px "JetBrains Mono", monospace';
              const tw = ctx.measureText(txt).width + 6;
              ctx.fillStyle = '#080B08';
              ctx.fillRect(mx - tw / 2, my - 7, tw, 14);
              ctx.strokeStyle = weight >= 0 ? '#39FF78' : '#285B37';
              ctx.lineWidth = 1;
              ctx.strokeRect(mx - tw / 2, my - 7, tw, 14);
              ctx.fillStyle = weight >= 0 ? '#39FF78' : '#8CBF98';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText(txt, mx, my);
            }
          }
        }
      }

      // 3. Отрисовка нейронов и их активаций
      for (let l = 0; l < numLayers; l++) {
        const layerNodes = positions[l];
        for (let n = 0; n < layerNodes.length; n++) {
          const { x, y, r } = layerNodes[n];
          const act = snapshot.activations[l]?.[n] ?? 0;
          const isHovered = hovered?.layerIdx === l && hovered?.neuronIdx === n;

          // Внешнее кольцо при наведении
          if (isHovered) {
            ctx.beginPath();
            ctx.arc(x, y, r + 5, 0, Math.PI * 2);
            ctx.strokeStyle = '#39FF78';
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.stroke();
            ctx.setLineDash([]);
          }

          // Фон узла
          ctx.beginPath();
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.fillStyle = '#0B110C';
          ctx.fill();

          // Внутренняя заливка уровня активации
          const normAct = clamp(Math.abs(act), 0, 1);
          ctx.beginPath();
          ctx.arc(x, y, Math.max(2, (r - 2) * (0.28 + normAct * 0.72)), 0, Math.PI * 2);
          if (act >= 0) {
            ctx.fillStyle = `rgba(57, 255, 120, ${(0.12 + normAct * 0.52).toFixed(3)})`;
          } else {
            ctx.fillStyle = `rgba(32, 68, 42, ${(0.25 + normAct * 0.5).toFixed(3)})`;
          }
          ctx.fill();

          // Контур узла
          ctx.beginPath();
          ctx.arc(x, y, r, 0, Math.PI * 2);
          ctx.lineWidth = isHovered ? 2 : 1.4;
          ctx.strokeStyle = act >= 0 ? '#39FF78' : '#285B37';
          ctx.stroke();

          // Значение активации внутри нейрона
          if (r >= 13) {
            ctx.font = `600 ${r >= 17 ? 9.5 : 8.5}px "JetBrains Mono", monospace`;
            ctx.fillStyle = '#D0E8D4';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(act.toFixed(2), x, y + 0.5);
          }

          // Боковые обозначения входов и выхода
          if (l === 0) {
            ctx.font = '700 10.5px "JetBrains Mono", monospace';
            ctx.fillStyle = '#8CBF98';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            const inLabel = snapshot.layerSizes[0] === 1 ? 'X' : `X${n + 1}`;
            ctx.fillText(inLabel, x - r - 8, y);
          } else if (l === numLayers - 1) {
            ctx.font = '700 10.5px "JetBrains Mono", monospace';
            ctx.fillStyle = '#39FF78';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText('Ŷ', x + r + 8, y);
          }
        }

        // Подписи слоёв снизу
        if (layerNodes.length > 0) {
          const colX = layerNodes[0].x;
          const label = snapshot.layerLabels[l] || `СЛОЙ ${l}`;
          const actName = snapshot.layerActivations[l] || '';

          ctx.textAlign = 'center';
          ctx.textBaseline = 'alphabetic';
          ctx.font = '700 10px "IBM Plex Mono", monospace';
          ctx.fillStyle = '#D0E8D4';
          ctx.fillText(label, colX, h - 25);

          ctx.font = '500 9.5px "JetBrains Mono", monospace';
          ctx.fillStyle = '#718575';
          ctx.fillText(
            `${layerNodes.length} нейр.${actName ? ` • ${actName.toUpperCase()}` : ''}`,
            colX,
            h - 11
          );
        }
      }

      ctx.restore();
      animFrameId = requestAnimationFrame(render);
    };

    animFrameId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animFrameId);
  }, [snapshot, canvasSize, signalEnabled, isTraining, hovered]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const layers = nodePositionsRef.current;
    for (let l = 0; l < layers.length; l++) {
      for (let n = 0; n < layers[l].length; n++) {
        const node = layers[l][n];
        const dist = Math.hypot(mx - node.x, my - node.y);
        if (dist <= node.r + 5) {
          setHovered({ layerIdx: l, neuronIdx: n, x: node.x, y: node.y });
          return;
        }
      }
    }
    if (hovered !== null) {
      setHovered(null);
    }
  };

  const hoveredActivation =
    hovered !== null ? snapshot.activations[hovered.layerIdx]?.[hovered.neuronIdx] ?? 0 : null;
  const hoveredBias =
    hovered !== null && hovered.layerIdx > 0
      ? snapshot.biases[hovered.layerIdx - 1]?.[hovered.neuronIdx] ?? 0
      : null;

  return (
    <div className="tech-panel np-visualizer-panel">
      <div className="tech-panel-header">
        <div className="tech-panel-title">
          <span className="tech-panel-code">[ГРАФ]</span>
          <span>{title}</span>
        </div>
        <div className="np-vis-actions">
          {onTriggerPulse && (
            <button
              type="button"
              className="retro-btn retro-btn-sm"
              onClick={onTriggerPulse}
              title="Подать случайный тестовый сигнал из выборки и пересчитать активации"
            >
              <PlayCircle size={13} />
              ТЕСТ СИГНАЛА
            </button>
          )}
          <button
            type="button"
            className={`retro-btn retro-btn-sm ${signalEnabled ? 'retro-btn-primary' : ''}`}
            onClick={() => setSignalEnabled((v) => !v)}
          >
            <Zap size={13} />
            ИМПУЛЬС: {signalEnabled ? 'ВКЛ' : 'ВЫКЛ'}
          </button>
        </div>
      </div>

      <div className="np-vis-canvas-wrap" ref={containerRef} style={{ height }}>
        <canvas
          ref={canvasRef}
          className="np-vis-canvas"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHovered(null)}
        />

        {hovered && (
          <div className="np-node-hud">
            <div className="np-node-hud-title">
              {snapshot.layerLabels[hovered.layerIdx]} // УЗЕЛ #{hovered.neuronIdx + 1}
            </div>
            <div className="np-node-hud-row">
              <span>Активация (a):</span>
              <strong className="phosphor-text">
                {hoveredActivation !== null ? hoveredActivation.toFixed(4) : '—'}
              </strong>
            </div>
            {hoveredBias !== null && (
              <div className="np-node-hud-row">
                <span>Смещение (b):</span>
                <strong>{hoveredBias.toFixed(4)}</strong>
              </div>
            )}
            {hovered.layerIdx > 0 && (
              <div className="np-node-hud-row">
                <span>Функция:</span>
                <strong>{snapshot.layerActivations[hovered.layerIdx].toUpperCase()}</strong>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="np-vis-legend">
        <div className="np-legend-items">
          <span className="np-legend-item">
            <span className="np-legend-swatch is-pos" />
            Положительный вес / активация (&gt; 0)
          </span>
          <span className="np-legend-item">
            <span className="np-legend-swatch is-neg" />
            Отрицательный вес / подавление (&lt; 0)
          </span>
          <span className="np-legend-item">
            <span className="np-legend-swatch is-thick" />
            Толщина линии = |вес связи|
          </span>
        </div>
        <div className="np-legend-probe">
          <Info size={12} />
          <span>
            Вход: [{snapshot.probeInput.map((v) => v.toFixed(2)).join(', ')}] → Выход Ŷ:{' '}
            <strong className="phosphor-text">
              {snapshot.probeOutput.map((v) => v.toFixed(3)).join(', ')}
            </strong>
          </span>
        </div>
      </div>
    </div>
  );
};
