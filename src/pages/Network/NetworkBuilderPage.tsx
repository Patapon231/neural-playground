import React from 'react';
import {
  Plus,
  Trash2,
  Minus,
  RotateCcw,
  Layers,
  Cpu,
  Sliders,
  Sparkles,
} from 'lucide-react';
import { NetworkVisualizer } from '../../components/NeuralNetwork/NetworkVisualizer';
import { StatCard } from '../../components/Statistics/StatCard';
import {
  ActivationType,
  HiddenLayerConfig,
  NetworkArchitecture,
  NetworkSnapshot,
  TaskType,
} from '../../types';
import {
  calculateLayerSizes,
  calculateTotalLayers,
  calculateTotalNeurons,
  calculateTotalParameters,
  getActivationLabel,
} from '../../utils/formatters';
import './NetworkBuilderPage.css';

interface NetworkBuilderPageProps {
  architecture: NetworkArchitecture;
  snapshot: NetworkSnapshot;
  taskType: TaskType;
  isTraining: boolean;
  onAddHiddenLayer: () => void;
  onRemoveHiddenLayer: (id: string) => void;
  onUpdateHiddenLayer: (id: string, patch: Partial<HiddenLayerConfig>) => void;
  onUpdateOutputActivation: (activation: ActivationType, auto: boolean) => void;
  onApplyPreset: (presetLayers: Array<{ neurons: number; activation: ActivationType }>) => void;
  onRebuildModel: () => void;
  onTriggerPulse: () => void;
}

const ACTIVATIONS: ActivationType[] = ['tanh', 'relu', 'sigmoid', 'elu', 'linear'];

const MAX_HIDDEN_LAYERS = 6;
const MIN_NEURONS = 1;
const MAX_NEURONS = 12;

export const NetworkBuilderPage: React.FC<NetworkBuilderPageProps> = ({
  architecture,
  snapshot,
  taskType,
  isTraining,
  onAddHiddenLayer,
  onRemoveHiddenLayer,
  onUpdateHiddenLayer,
  onUpdateOutputActivation,
  onApplyPreset,
  onRebuildModel,
  onTriggerPulse,
}) => {
  const totalLayers = calculateTotalLayers(architecture);
  const totalNeurons = calculateTotalNeurons(architecture);
  const totalParams = calculateTotalParameters(architecture);
  const layerSizes = calculateLayerSizes(architecture);

  return (
    <div className="np-builder-page">
      <div className="page-header-bar">
        <div>
          <h1 className="page-title">02 // КОНСТРУКТОР АРХИТЕКТУРЫ НЕЙРОСЕТИ</h1>
          <p className="page-subtitle">
            Проектирование топологии слоёв, количества нейронов и нелинейных функций активации TensorFlow.js
          </p>
        </div>

        <div className="np-builder-top-btns">
          <button
            type="button"
            className="retro-btn"
            onClick={onRebuildModel}
            disabled={isTraining}
          >
            <RotateCcw size={14} />
            ПЕРЕИНИЦИАЛИЗИРОВАТЬ ВЕСА
          </button>
          <button
            type="button"
            className="retro-btn retro-btn-primary"
            onClick={onAddHiddenLayer}
            disabled={isTraining || architecture.hiddenLayers.length >= MAX_HIDDEN_LAYERS}
          >
            <Plus size={14} />
            ДОБАВИТЬ СКРЫТЫЙ СЛОЙ ({architecture.hiddenLayers.length}/{MAX_HIDDEN_LAYERS})
          </button>
        </div>
      </div>

      <div className="np-stat-grid">
        <StatCard
          code="ARC.01"
          label="Всего слоёв"
          value={totalLayers}
          subtext={`${architecture.hiddenLayers.length} скрытых полносвязных Dense`}
          icon={<Layers size={15} />}
        />
        <StatCard
          code="ARC.02"
          label="Узлы (нейроны)"
          value={totalNeurons}
          subtext={`Вход: ${architecture.inputNeurons} • Выход: ${architecture.outputNeurons}`}
          icon={<Cpu size={15} />}
        />
        <StatCard
          code="ARC.03"
          label="Обучаемые параметры"
          value={totalParams}
          subtext="Сумма матриц весов и векторов смещений"
          accent
          icon={<Sliders size={15} />}
        />
        <StatCard
          code="ARC.04"
          label="Режим выхода"
          value={architecture.outputActivation.toUpperCase()}
          subtext={
            taskType === 'classification'
              ? 'Задача: Классификация'
              : 'Задача: Линейная регрессия'
          }
          icon={<Sparkles size={15} />}
        />
      </div>

      {/* Быстрые пресеты архитектур */}
      <div className="tech-panel np-presets-bar">
        <span className="np-presets-label">ТИПОВЫЕ ТОПОЛОГИИ:</span>
        <div className="np-presets-buttons">
          <button
            type="button"
            className="retro-btn retro-btn-sm"
            disabled={isTraining}
            onClick={() => onApplyPreset([])}
          >
            ЛИНЕЙНЫЙ ПЕРЦЕПТРОН [0 СКР.]
          </button>
          <button
            type="button"
            className="retro-btn retro-btn-sm"
            disabled={isTraining}
            onClick={() => onApplyPreset([{ neurons: 4, activation: 'tanh' }])}
          >
            БАЗОВАЯ MLP [4]
          </button>
          <button
            type="button"
            className="retro-btn retro-btn-sm"
            disabled={isTraining}
            onClick={() =>
              onApplyPreset([
                { neurons: 6, activation: 'tanh' },
                { neurons: 4, activation: 'tanh' },
              ])
            }
          >
            КЛАССИЧЕСКАЯ [6 → 4]
          </button>
          <button
            type="button"
            className="retro-btn retro-btn-sm"
            disabled={isTraining}
            onClick={() =>
              onApplyPreset([
                { neurons: 8, activation: 'relu' },
                { neurons: 6, activation: 'tanh' },
                { neurons: 4, activation: 'tanh' },
              ])
            }
          >
            ГЛУБОКАЯ СЕТЬ [8 → 6 → 4]
          </button>
        </div>
      </div>

      <div className="np-builder-grid">
        {/* Левая колонка: управление слоями */}
        <div className="np-layers-editor">
          {/* Входной слой */}
          <div className="tech-panel np-layer-card is-fixed">
            <div className="np-layer-card-head">
              <span className="tech-badge">ВХОДНОЙ СЛОЙ [INPUT]</span>
              <span className="np-layer-dim">{architecture.inputNeurons} НЕЙРОНА(ОВ)</span>
            </div>
            <p className="np-layer-desc">
              Размерность входа определяется выбранным набором данных (
              {architecture.inputNeurons === 2
                ? 'координаты признаков X₁ и X₂ на плоскости'
                : 'входной признак X для линейной зависимости'}
              ).
            </p>
          </div>

          {/* Скрытые слои */}
          {architecture.hiddenLayers.length === 0 ? (
            <div className="tech-panel np-empty-hidden">
              <p>
                Скрытые слои отсутствуют — сеть работает как линейный перцептрон и способна
                разделять только линейно разделимые выборки.
              </p>
              <button
                type="button"
                className="retro-btn retro-btn-primary retro-btn-sm"
                onClick={onAddHiddenLayer}
                disabled={isTraining}
              >
                <Plus size={13} />
                ДОБАВИТЬ ПЕРВЫЙ СКРЫТЫЙ СЛОЙ
              </button>
            </div>
          ) : (
            architecture.hiddenLayers.map((layer, idx) => {
              const prevCount = idx === 0 ? architecture.inputNeurons : architecture.hiddenLayers[idx - 1].neurons;
              const layerParams = prevCount * layer.neurons + layer.neurons;

              return (
                <div key={layer.id} className="tech-panel np-layer-card">
                  <div className="np-layer-card-head">
                    <div className="np-layer-head-left">
                      <span className="tech-badge tech-badge-active">
                        СКРЫТЫЙ СЛОЙ #{idx + 1}
                      </span>
                      <span className="np-layer-param-tag">{layerParams} парам.</span>
                    </div>
                    <button
                      type="button"
                      className="retro-btn retro-btn-danger retro-btn-sm"
                      onClick={() => onRemoveHiddenLayer(layer.id)}
                      disabled={isTraining}
                      title="Удалить скрытый слой"
                    >
                      <Trash2 size={12} />
                      УДАЛИТЬ
                    </button>
                  </div>

                  <div className="np-layer-controls">
                    <div className="np-control-group">
                      <div className="np-control-label-row">
                        <span>Количество нейронов:</span>
                        <strong className="phosphor-text">{layer.neurons}</strong>
                      </div>
                      <div className="np-neuron-stepper">
                        <button
                          type="button"
                          className="retro-btn retro-btn-sm"
                          disabled={isTraining || layer.neurons <= MIN_NEURONS}
                          onClick={() =>
                            onUpdateHiddenLayer(layer.id, {
                              neurons: Math.max(MIN_NEURONS, layer.neurons - 1),
                            })
                          }
                        >
                          <Minus size={12} />
                        </button>
                        <input
                          type="range"
                          className="retro-range"
                          min={MIN_NEURONS}
                          max={MAX_NEURONS}
                          value={layer.neurons}
                          disabled={isTraining}
                          onChange={(e) =>
                            onUpdateHiddenLayer(layer.id, {
                              neurons: Number(e.target.value),
                            })
                          }
                        />
                        <button
                          type="button"
                          className="retro-btn retro-btn-sm"
                          disabled={isTraining || layer.neurons >= MAX_NEURONS}
                          onClick={() =>
                            onUpdateHiddenLayer(layer.id, {
                              neurons: Math.min(MAX_NEURONS, layer.neurons + 1),
                            })
                          }
                        >
                          <Plus size={12} />
                        </button>
                      </div>
                    </div>

                    <div className="np-control-group">
                      <label className="np-control-label-row">
                        <span>Функция активации:</span>
                      </label>
                      <select
                        className="retro-select np-full-select"
                        value={layer.activation}
                        disabled={isTraining}
                        onChange={(e) =>
                          onUpdateHiddenLayer(layer.id, {
                            activation: e.target.value as ActivationType,
                          })
                        }
                      >
                        {ACTIVATIONS.map((act) => (
                          <option key={act} value={act}>
                            {getActivationLabel(act)}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
              );
            })
          )}

          {/* Выходной слой */}
          <div className="tech-panel np-layer-card is-output">
            <div className="np-layer-card-head">
              <span className="tech-badge">ВЫХОДНОЙ СЛОЙ [OUTPUT Ŷ]</span>
              <span className="np-layer-dim">
                {layerSizes[layerSizes.length - 2] * architecture.outputNeurons +
                  architecture.outputNeurons}{' '}
                парам.
              </span>
            </div>

            <div className="np-layer-controls">
              <label className="np-checkbox-row">
                <input
                  type="checkbox"
                  checked={architecture.autoOutputActivation}
                  disabled={isTraining}
                  onChange={(e) => {
                    const auto = e.target.checked;
                    const recommended: ActivationType =
                      taskType === 'classification' ? 'sigmoid' : 'linear';
                    onUpdateOutputActivation(
                      auto ? recommended : architecture.outputActivation,
                      auto
                    );
                  }}
                />
                <span>Автоподбор активации выхода под выбранную задачу</span>
              </label>

              <div className="np-control-group">
                <label className="np-control-label-row">
                  <span>Активация выходного нейрона:</span>
                </label>
                <select
                  className="retro-select np-full-select"
                  value={architecture.outputActivation}
                  disabled={isTraining || architecture.autoOutputActivation}
                  onChange={(e) =>
                    onUpdateOutputActivation(e.target.value as ActivationType, false)
                  }
                >
                  {ACTIVATIONS.map((act) => (
                    <option key={act} value={act}>
                      {getActivationLabel(act)}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Правая колонка: интерактивный граф и таблица параметров */}
        <div className="np-builder-preview">
          <NetworkVisualizer
            snapshot={snapshot}
            isTraining={isTraining}
            height={415}
            onTriggerPulse={onTriggerPulse}
            title="ИНТЕРАКТИВНАЯ СХЕМА СОЗДАВАЕМОЙ СЕТИ"
          />

          <div className="tech-panel np-param-table-panel">
            <div className="tech-panel-header">
              <div className="tech-panel-title">
                <span className="tech-panel-code">[МАТРИЦА]</span>
                <span>РАСЧЁТ ПАРАМЕТРОВ ПО СЛОЯМ</span>
              </div>
              <span className="phosphor-text">ИТОГО: {totalParams}</span>
            </div>

            <div className="np-param-table-wrap">
              <table className="np-tech-table">
                <thead>
                  <tr>
                    <th>Слой</th>
                    <th>Форма (Вх → Вых)</th>
                    <th>Активация</th>
                    <th>Веса (W)</th>
                    <th>Смещения (b)</th>
                    <th>Всего</th>
                  </tr>
                </thead>
                <tbody>
                  {layerSizes.slice(1).map((currSize, idx) => {
                    const prevSize = layerSizes[idx];
                    const wCount = prevSize * currSize;
                    const bCount = currSize;
                    const isOut = idx === layerSizes.length - 2;
                    const act = isOut
                      ? architecture.outputActivation
                      : architecture.hiddenLayers[idx].activation;

                    return (
                      <tr key={idx}>
                        <td>{isOut ? 'Выходной (Dense)' : `Скрытый #${idx + 1} (Dense)`}</td>
                        <td>
                          {prevSize} × {currSize}
                        </td>
                        <td>{act.toUpperCase()}</td>
                        <td>{wCount}</td>
                        <td>{bCount}</td>
                        <td className="phosphor-text">{wCount + bCount}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
