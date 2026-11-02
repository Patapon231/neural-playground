import { useCallback, useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import {
  ActivationType,
  DatasetConfig,
  DatasetId,
  EpochMetric,
  GeneratedDataset,
  HiddenLayerConfig,
  LogCategory,
  NetworkArchitecture,
  NetworkSnapshot,
  PredictionVisualData,
  TerminalLogEntry,
  TrainingConfig,
  TrainingRunSummary,
  TrainingStatus,
} from '../types';
import { DATASETS_CATALOG, generateDataset } from '../services/datasets';
import {
  buildSequentialModel,
  computePredictionVisuals,
  createFallbackSnapshot,
  evaluateDatasetMetrics,
  extractNetworkSnapshot,
  getTensorMemoryStats,
  initTensorFlowEngine,
} from '../services/neuralNetwork';
import { runModelTraining } from '../services/training';
import {
  calculateTotalLayers,
  calculateTotalNeurons,
  calculateTotalParameters,
  formatArchitectureString,
  formatTimestamp,
  getOptimizerLabel,
  getTrainingStatusLabel,
} from '../utils/formatters';

const DEFAULT_ARCHITECTURE: NetworkArchitecture = {
  inputNeurons: 2,
  hiddenLayers: [
    { id: 'layer-1', neurons: 6, activation: 'tanh' },
    { id: 'layer-2', neurons: 4, activation: 'tanh' },
  ],
  outputNeurons: 1,
  outputActivation: 'sigmoid',
  autoOutputActivation: true,
};

const DEFAULT_DATASET_CONFIG: DatasetConfig = {
  id: 'xor',
  sampleCount: 200,
  noise: 0.06,
};

const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  epochs: 100,
  learningRate: 0.03,
  batchSize: 32,
  optimizer: 'adam',
};

export function usePlayground() {
  const [systemReady, setSystemReady] = useState<boolean>(false);
  const [tfBackend, setTfBackend] = useState<string>('инициализация');
  const [tensorCount, setTensorCount] = useState<number>(0);

  const [architecture, setArchitecture] = useState<NetworkArchitecture>(DEFAULT_ARCHITECTURE);
  const [dataset, setDataset] = useState<GeneratedDataset>(() =>
    generateDataset(DEFAULT_DATASET_CONFIG)
  );
  const [trainingConfig, setTrainingConfig] =
    useState<TrainingConfig>(DEFAULT_TRAINING_CONFIG);

  const [modelParamsCount, setModelParamsCount] = useState<number>(() =>
    calculateTotalParameters(DEFAULT_ARCHITECTURE)
  );
  const [snapshot, setSnapshot] = useState<NetworkSnapshot>(() =>
    createFallbackSnapshot(DEFAULT_ARCHITECTURE)
  );
  const [predictionVisuals, setPredictionVisuals] =
    useState<PredictionVisualData | null>(null);

  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>('idle');
  const [currentEpoch, setCurrentEpoch] = useState<number>(0);
  const [currentLoss, setCurrentLoss] = useState<number | null>(null);
  const [currentAccuracy, setCurrentAccuracy] = useState<number | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number>(0);

  const [liveHistory, setLiveHistory] = useState<EpochMetric[]>([]);
  const [runsHistory, setRunsHistory] = useState<TrainingRunSummary[]>([]);
  const [logs, setLogs] = useState<TerminalLogEntry[]>([]);

  const modelRef = useRef<tf.Sequential | null>(null);
  const probeRef = useRef<number[]>([0.55, -0.5]);
  const stopFlagRef = useRef<boolean>(false);
  const runCounterRef = useRef<number>(0);

  const addLog = useCallback(
    (category: LogCategory, message: string, isError = false) => {
      setLogs((prev) => {
        const next: TerminalLogEntry = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          timestamp: formatTimestamp(),
          category,
          message,
          isError,
        };
        const updated = [...prev, next];
        return updated.length > 250 ? updated.slice(updated.length - 250) : updated;
      });
    },
    []
  );

  const refreshMemoryTelemetry = useCallback(() => {
    try {
      const mem = getTensorMemoryStats();
      setTensorCount(mem.numTensors);
    } catch {
      // Игнорируем до полной готовности бэкенда
    }
  }, []);

  const syncVisualsFromModel = useCallback(
    (
      model: tf.Sequential,
      arch: NetworkArchitecture,
      ds: GeneratedDataset,
      customProbe?: number[]
    ) => {
      const defaultProbe =
        ds.points.length > 0
          ? ds.meta.inputDim === 2
            ? [ds.points[0].x, ds.points[0].y]
            : [ds.points[0].x]
          : ds.meta.inputDim === 2
            ? [0.5, -0.5]
            : [0.5];

      const validProbe =
        customProbe && customProbe.length === arch.inputNeurons
          ? customProbe
          : probeRef.current.length === arch.inputNeurons
            ? probeRef.current
            : defaultProbe;

      probeRef.current = validProbe;

      const snap = extractNetworkSnapshot(model, arch, validProbe);
      const visuals = computePredictionVisuals(model, ds.meta.taskType, ds.meta.inputDim);
      setSnapshot(snap);
      setPredictionVisuals(visuals);
      refreshMemoryTelemetry();
    },
    [refreshMemoryTelemetry]
  );

  const instantiateModel = useCallback(
    (
      arch: NetworkArchitecture,
      trnCfg: TrainingConfig,
      ds: GeneratedDataset,
      logMessage?: string
    ) => {
      try {
        if (modelRef.current) {
          modelRef.current.dispose();
          modelRef.current = null;
        }

        const { model, paramCount } = buildSequentialModel(
          arch,
          trnCfg,
          ds.meta.taskType
        );
        modelRef.current = model;
        setModelParamsCount(paramCount);

        const initialEval = evaluateDatasetMetrics(
          model,
          ds.points,
          ds.meta.taskType,
          ds.meta.inputDim
        );

        setCurrentEpoch(0);
        setElapsedMs(0);
        setCurrentLoss(initialEval.loss);
        setCurrentAccuracy(initialEval.accuracy);
        setTrainingStatus('idle');
        setLiveHistory([]);

        syncVisualsFromModel(model, arch, ds);

        if (logMessage) {
          addLog('МОДЕЛЬ', `${logMessage} (${paramCount} параметров).`);
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setTrainingStatus('error');
        addLog('ОШИБКА', `Сбой сборки модели TensorFlow.js: ${msg}`, true);
      }
    },
    [addLog, syncVisualsFromModel]
  );

  // Первичная инициализация TensorFlow.js при монтировании
  useEffect(() => {
    let cancelled = false;

    (async () => {
      addLog('СИСТЕМА', 'Запуск лабораторного комплекса NEURAL PLAYGROUND v1.04...');
      try {
        const engine = await initTensorFlowEngine();
        if (cancelled) return;
        setTfBackend(engine.backend);
        setSystemReady(true);

        addLog(
          'СИСТЕМА',
          `Вычислительное ядро TensorFlow.js готово. Активный ускоритель: ${engine.backend.toUpperCase()}.`
        );
        addLog(
          'ДАННЫЕ',
          `Загружен набор данных «${dataset.meta.name}» (${dataset.points.length} точек).`
        );

        instantiateModel(
          DEFAULT_ARCHITECTURE,
          DEFAULT_TRAINING_CONFIG,
          dataset,
          `Инициализирована нейронная сеть ${formatArchitectureString(DEFAULT_ARCHITECTURE)}`
        );
      } catch (err) {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : String(err);
        setTrainingStatus('error');
        addLog('ОШИБКА', `Ошибка инициализации TensorFlow.js: ${msg}`, true);
      }
    })();

    return () => {
      cancelled = true;
      if (modelRef.current) {
        modelRef.current.dispose();
        modelRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Конструктор архитектуры: добавление скрытого слоя
  const addHiddenLayer = useCallback(() => {
    if (trainingStatus === 'training') return;
    const nextLayer: HiddenLayerConfig = {
      id: `layer-${Date.now()}-${Math.random().toString(36).slice(2, 5)}`,
      neurons: 4,
      activation: 'tanh',
    };
    const nextArch: NetworkArchitecture = {
      ...architecture,
      hiddenLayers: [...architecture.hiddenLayers, nextLayer],
    };
    setArchitecture(nextArch);
    instantiateModel(
      nextArch,
      trainingConfig,
      dataset,
      `Добавлен скрытый слой #${nextArch.hiddenLayers.length} (4 нейрона, TANH)`
    );
  }, [architecture, dataset, instantiateModel, trainingConfig, trainingStatus]);

  const removeHiddenLayer = useCallback(
    (id: string) => {
      if (trainingStatus === 'training') return;
      const nextArch: NetworkArchitecture = {
        ...architecture,
        hiddenLayers: architecture.hiddenLayers.filter((l) => l.id !== id),
      };
      setArchitecture(nextArch);
      instantiateModel(
        nextArch,
        trainingConfig,
        dataset,
        `Удалён скрытый слой. Новая схема: ${formatArchitectureString(nextArch)}`
      );
    },
    [architecture, dataset, instantiateModel, trainingConfig, trainingStatus]
  );

  const updateHiddenLayer = useCallback(
    (id: string, patch: Partial<HiddenLayerConfig>) => {
      if (trainingStatus === 'training') return;
      const nextArch: NetworkArchitecture = {
        ...architecture,
        hiddenLayers: architecture.hiddenLayers.map((l) =>
          l.id === id ? { ...l, ...patch } : l
        ),
      };
      setArchitecture(nextArch);
      instantiateModel(
        nextArch,
        trainingConfig,
        dataset,
        `Обновлена конфигурация слоя: ${formatArchitectureString(nextArch)}`
      );
    },
    [architecture, dataset, instantiateModel, trainingConfig, trainingStatus]
  );

  const updateOutputActivation = useCallback(
    (activation: ActivationType, auto: boolean) => {
      if (trainingStatus === 'training') return;
      const nextArch: NetworkArchitecture = {
        ...architecture,
        outputActivation: activation,
        autoOutputActivation: auto,
      };
      setArchitecture(nextArch);
      instantiateModel(
        nextArch,
        trainingConfig,
        dataset,
        `Функция активации выхода изменена на ${activation.toUpperCase()}`
      );
    },
    [architecture, dataset, instantiateModel, trainingConfig, trainingStatus]
  );

  const applyArchitecturePreset = useCallback(
    (presetLayers: Array<{ neurons: number; activation: ActivationType }>) => {
      if (trainingStatus === 'training') return;
      const nextArch: NetworkArchitecture = {
        ...architecture,
        hiddenLayers: presetLayers.map((p, idx) => ({
          id: `preset-${Date.now()}-${idx}`,
          neurons: p.neurons,
          activation: p.activation,
        })),
      };
      setArchitecture(nextArch);
      instantiateModel(
        nextArch,
        trainingConfig,
        dataset,
        `Применена типовая архитектура: ${formatArchitectureString(nextArch)}`
      );
    },
    [architecture, dataset, instantiateModel, trainingConfig, trainingStatus]
  );

  const resetModel = useCallback(() => {
    if (trainingStatus === 'training') return;
    instantiateModel(
      architecture,
      trainingConfig,
      dataset,
      'Веса нейронной сети сброшены и переинициализированы'
    );
  }, [architecture, dataset, instantiateModel, trainingConfig, trainingStatus]);

  // Выбор и настройка набора данных
  const selectDataset = useCallback(
    (id: DatasetId) => {
      if (trainingStatus === 'training') return;
      const targetMeta = DATASETS_CATALOG[id];
      const nextDs = generateDataset({
        ...dataset.config,
        id,
      });
      setDataset(nextDs);

      const nextOutputAct: ActivationType = architecture.autoOutputActivation
        ? targetMeta.taskType === 'classification'
          ? 'sigmoid'
          : 'linear'
        : architecture.outputActivation;

      const nextArch: NetworkArchitecture = {
        ...architecture,
        inputNeurons: targetMeta.inputDim,
        outputActivation: nextOutputAct,
      };
      setArchitecture(nextArch);

      addLog(
        'ДАННЫЕ',
        `Выбран набор данных «${nextDs.meta.name}» (${nextDs.points.length} точек, шум ${Math.round(nextDs.config.noise * 100)}%).`
      );

      instantiateModel(
        nextArch,
        trainingConfig,
        nextDs,
        `Модель адаптирована под набор «${nextDs.meta.code}»`
      );
    },
    [addLog, architecture, dataset.config, instantiateModel, trainingConfig, trainingStatus]
  );

  const updateDatasetConfig = useCallback(
    (patch: Partial<DatasetConfig>) => {
      if (trainingStatus === 'training') return;
      const nextCfg: DatasetConfig = { ...dataset.config, ...patch };
      const nextDs = generateDataset(nextCfg);
      setDataset(nextDs);

      if (modelRef.current) {
        const evalRes = evaluateDatasetMetrics(
          modelRef.current,
          nextDs.points,
          nextDs.meta.taskType,
          nextDs.meta.inputDim
        );
        setCurrentLoss(evalRes.loss);
        setCurrentAccuracy(evalRes.accuracy);
        syncVisualsFromModel(modelRef.current, architecture, nextDs);
      }

      addLog(
        'ДАННЫЕ',
        `Обновлены параметры выборки «${nextDs.meta.code}»: ${nextDs.points.length} точек, шум ${Math.round(nextDs.config.noise * 100)}%.`
      );
    },
    [addLog, architecture, dataset.config, syncVisualsFromModel, trainingStatus]
  );

  const regenerateCurrentDataset = useCallback(() => {
    if (trainingStatus === 'training') return;
    const nextDs = generateDataset(dataset.config);
    setDataset(nextDs);
    if (modelRef.current) {
      const evalRes = evaluateDatasetMetrics(
        modelRef.current,
        nextDs.points,
        nextDs.meta.taskType,
        nextDs.meta.inputDim
      );
      setCurrentLoss(evalRes.loss);
      setCurrentAccuracy(evalRes.accuracy);
      syncVisualsFromModel(modelRef.current, architecture, nextDs);
    }
    addLog('ДАННЫЕ', `Сгенерирована новая случайная реализация «${nextDs.meta.name}».`);
  }, [addLog, architecture, dataset.config, syncVisualsFromModel, trainingStatus]);

  const updateTrainingConfig = useCallback((patch: Partial<TrainingConfig>) => {
    setTrainingConfig((prev) => ({ ...prev, ...patch }));
  }, []);

  // Зондирование сети конкретным вектором признаков (при клике на 2D график или кнопке Тест сигнала)
  const probeNetwork = useCallback(
    (inputVec: number[]) => {
      if (!modelRef.current) return;
      probeRef.current = inputVec;
      const snap = extractNetworkSnapshot(modelRef.current, architecture, inputVec);
      setSnapshot(snap);
      addLog(
        'МОДЕЛЬ',
        `Прямой проход сигнала [${inputVec.map((v) => v.toFixed(2)).join(', ')}] → Ŷ = ${snap.probeOutput.map((v) => v.toFixed(4)).join(', ')}`
      );
    },
    [addLog, architecture]
  );

  const triggerRandomPulse = useCallback(() => {
    if (!modelRef.current || dataset.points.length === 0) return;
    const randomPt = dataset.points[Math.floor(Math.random() * dataset.points.length)];
    const vec =
      dataset.meta.inputDim === 2 ? [randomPt.x, randomPt.y] : [randomPt.x];
    probeNetwork(vec);
  }, [dataset.meta.inputDim, dataset.points, probeNetwork]);

  // Запуск и остановка настоящего обучения TensorFlow.js
  const startTraining = useCallback(async () => {
    if (!systemReady || trainingStatus === 'training') return;

    if (!modelRef.current) {
      instantiateModel(architecture, trainingConfig, dataset);
    }
    const model = modelRef.current;
    if (!model) return;

    stopFlagRef.current = false;
    setTrainingStatus('training');
    setCurrentEpoch(0);
    setElapsedMs(0);
    setLiveHistory([]);

    const runNum = runCounterRef.current + 1;
    runCounterRef.current = runNum;
    const startedStamp = formatTimestamp().slice(0, 8);

    addLog(
      'ОБУЧЕНИЕ',
      `Запуск цикла #${runNum}: выборка=${dataset.meta.code}, эпох=${trainingConfig.epochs}, LR=${trainingConfig.learningRate}, опт=${trainingConfig.optimizer.toUpperCase()}.`
    );

    const visualStep = trainingConfig.epochs > 140 ? 2 : 1;
    const logStep = Math.max(10, Math.round(trainingConfig.epochs / 4));

    try {
      const result = await runModelTraining(
        model,
        architecture,
        dataset.points,
        dataset.meta.taskType,
        dataset.meta.inputDim,
        trainingConfig,
        {
          shouldStop: () => stopFlagRef.current,
          onEpochUpdate: (metric) => {
            setCurrentEpoch(metric.epoch);
            setCurrentLoss(metric.loss);
            setCurrentAccuracy(metric.accuracy);
            setElapsedMs(metric.elapsedMs);
            setLiveHistory((prev) => [...prev, metric]);

            if (
              metric.epoch % visualStep === 0 ||
              metric.epoch === trainingConfig.epochs
            ) {
              const sampleIdx = metric.epoch % Math.max(1, dataset.points.length);
              const pt = dataset.points[sampleIdx];
              const liveProbe =
                pt !== undefined
                  ? dataset.meta.inputDim === 2
                    ? [pt.x, pt.y]
                    : [pt.x]
                  : probeRef.current;

              syncVisualsFromModel(model, architecture, dataset, liveProbe);
            }

            if (metric.epoch % logStep === 0 && metric.epoch < trainingConfig.epochs) {
              addLog(
                'ОБУЧЕНИЕ',
                `Эпоха ${metric.epoch}/${trainingConfig.epochs} • Loss=${metric.loss.toFixed(5)} • Точность=${(metric.accuracy * 100).toFixed(1)}%`
              );
            }
          },
        }
      );

      syncVisualsFromModel(model, architecture, dataset);

      const finalStatus = result.stoppedEarly ? 'stopped' : 'completed';
      setTrainingStatus(finalStatus);

      if (result.history.length > 0) {
        const first = result.history[0];
        const last = result.history[result.history.length - 1];

        const summary: TrainingRunSummary = {
          id: `run-${Date.now()}`,
          runNumber: runNum,
          startedAt: startedStamp,
          datasetId: dataset.meta.id,
          datasetName: dataset.meta.name,
          taskType: dataset.meta.taskType,
          architectureSummary: formatArchitectureString(architecture),
          totalParams: model.countParams(),
          epochsCompleted: result.completedEpochs,
          targetEpochs: trainingConfig.epochs,
          learningRate: trainingConfig.learningRate,
          optimizer: trainingConfig.optimizer,
          initialLoss: first.loss,
          finalLoss: last.loss,
          initialAccuracy: first.accuracy,
          finalAccuracy: last.accuracy,
          durationMs: result.durationMs,
          status: finalStatus,
          history: result.history,
        };

        setRunsHistory((prev) => [...prev, summary]);

        if (result.stoppedEarly) {
          addLog(
            'ОБУЧЕНИЕ',
            `Обучение #${runNum} остановлено на эпохе ${result.completedEpochs}/${trainingConfig.epochs}. Loss=${last.loss.toFixed(5)}, Точность=${(last.accuracy * 100).toFixed(1)}%.`
          );
        } else {
          addLog(
            'ОБУЧЕНИЕ',
            `Обучение #${runNum} успешно завершено (${result.completedEpochs} эпох за ${(result.durationMs / 1000).toFixed(2)} с). Финальный Loss=${last.loss.toFixed(5)}, Точность=${(last.accuracy * 100).toFixed(1)}%.`
          );
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setTrainingStatus('error');
      addLog('ОШИБКА', `Ошибка во время обучения модели: ${msg}`, true);
    }
  }, [
    addLog,
    architecture,
    dataset,
    instantiateModel,
    syncVisualsFromModel,
    systemReady,
    trainingConfig,
    trainingStatus,
  ]);

  const stopTraining = useCallback(() => {
    if (trainingStatus !== 'training') return;
    stopFlagRef.current = true;
    addLog('ОБУЧЕНИЕ', 'Передан сигнал прерывания процесса обучения...');
  }, [addLog, trainingStatus]);

  const clearLogs = useCallback(() => {
    setLogs([]);
  }, []);

  // Командный интерпретатор терминала
  const executeCommand = useCallback(
    (rawCommand: string) => {
      addLog('КОМАНДА', `NP-88> ${rawCommand}`);
      const parts = rawCommand.trim().split(/\s+/);
      const cmd = (parts[0] || '').toLowerCase();
      const arg = (parts[1] || '').toLowerCase();

      switch (cmd) {
        case 'help': {
          addLog(
            'СИСТЕМА',
            [
              'Доступные команды лабораторного терминала:',
              '  help                  — показать список команд',
              '  status                — состояние системы, памяти TF.js и обучения',
              '  info                  — спецификация архитектуры сети и набора данных',
              '  train                 — запустить цикл обучения модели',
              '  stop                  — остановить активное обучение',
              '  reset                 — переинициализировать веса модели',
              '  dataset <код>         — выбрать данные (xor | circles | linear | clusters)',
              '  clear                 — очистить журнал консоли',
            ].join('\n')
          );
          break;
        }
        case 'clear': {
          setLogs([]);
          break;
        }
        case 'status': {
          const mem = getTensorMemoryStats();
          const kb = (mem.numBytes / 1024).toFixed(1);
          addLog(
            'СИСТЕМА',
            `Ядро: ${tfBackend.toUpperCase()} | Тензоров: ${mem.numTensors} (${kb} КБ) | Статус: ${getTrainingStatusLabel(trainingStatus)} | Эпоха: ${currentEpoch}/${trainingConfig.epochs} | Loss: ${currentLoss !== null ? currentLoss.toFixed(5) : '—'} | Точность: ${currentAccuracy !== null ? `${(currentAccuracy * 100).toFixed(1)}%` : '—'}`
          );
          break;
        }
        case 'info': {
          addLog(
            'МОДЕЛЬ',
            `Слоёв: ${calculateTotalLayers(architecture)} | Нейронов: ${calculateTotalNeurons(architecture)} | Параметров: ${modelParamsCount} | Схема: ${formatArchitectureString(architecture)} | Выборка: ${dataset.meta.name} (${dataset.points.length} точек) | Оптимизатор: ${getOptimizerLabel(trainingConfig.optimizer)}, LR=${trainingConfig.learningRate}`
          );
          break;
        }
        case 'train': {
          if (trainingStatus === 'training') {
            addLog('СИСТЕМА', 'Обучение уже выполняется.');
          } else {
            void startTraining();
          }
          break;
        }
        case 'stop': {
          if (trainingStatus === 'training') {
            stopTraining();
          } else {
            addLog('СИСТЕМА', 'Активный процесс обучения отсутствует.');
          }
          break;
        }
        case 'reset': {
          if (trainingStatus === 'training') {
            addLog('ОШИБКА', 'Остановите обучение перед сбросом весов модели.', true);
          } else {
            resetModel();
          }
          break;
        }
        case 'dataset': {
          if (arg === 'xor' || arg === 'circles' || arg === 'linear' || arg === 'clusters') {
            selectDataset(arg);
          } else {
            addLog(
              'ОШИБКА',
              'Укажите корректный код выборки: dataset xor | circles | linear | clusters',
              true
            );
          }
          break;
        }
        default: {
          addLog(
            'ОШИБКА',
            `Неизвестная директива "${cmd}". Введите "help" для просмотра справки.`,
            true
          );
        }
      }
    },
    [
      addLog,
      architecture,
      currentAccuracy,
      currentEpoch,
      currentLoss,
      dataset.meta.name,
      dataset.points.length,
      modelParamsCount,
      resetModel,
      selectDataset,
      startTraining,
      stopTraining,
      tfBackend,
      trainingConfig.epochs,
      trainingConfig.learningRate,
      trainingConfig.optimizer,
      trainingStatus,
    ]
  );

  return {
    systemReady,
    tfBackend,
    tensorCount,
    architecture,
    modelParamsCount,
    snapshot,
    dataset,
    predictionVisuals,
    trainingConfig,
    trainingStatus,
    currentEpoch,
    currentLoss,
    currentAccuracy,
    elapsedMs,
    liveHistory,
    runsHistory,
    logs,
    addHiddenLayer,
    removeHiddenLayer,
    updateHiddenLayer,
    updateOutputActivation,
    applyArchitecturePreset,
    resetModel,
    selectDataset,
    updateDatasetConfig,
    regenerateCurrentDataset,
    updateTrainingConfig,
    probeNetwork,
    triggerRandomPulse,
    startTraining,
    stopTraining,
    clearLogs,
    executeCommand,
  };
}
