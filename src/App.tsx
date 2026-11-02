import React, { useState } from 'react';
import { Header } from './components/Header/Header';
import { Sidebar } from './components/Sidebar/Sidebar';
import { Terminal } from './components/Terminal/Terminal';
import { DashboardPage } from './pages/Dashboard/DashboardPage';
import { NetworkBuilderPage } from './pages/Network/NetworkBuilderPage';
import { DatasetsPage } from './pages/Datasets/DatasetsPage';
import { TrainingPage } from './pages/Training/TrainingPage';
import { AnalyticsPage } from './pages/Analytics/AnalyticsPage';
import { usePlayground } from './hooks/usePlayground';
import { PageId } from './types';
import { calculateTotalLayers } from './utils/formatters';

export const App: React.FC = () => {
  const [activePage, setActivePage] = useState<PageId>('dashboard');
  const [crtEnabled, setCrtEnabled] = useState<boolean>(true);
  const [terminalOpen, setTerminalOpen] = useState<boolean>(true);

  const lab = usePlayground();

  const renderActivePage = () => {
    switch (activePage) {
      case 'dashboard':
        return (
          <DashboardPage
            architecture={lab.architecture}
            modelParamsCount={lab.modelParamsCount}
            snapshot={lab.snapshot}
            dataset={lab.dataset}
            trainingConfig={lab.trainingConfig}
            trainingStatus={lab.trainingStatus}
            currentEpoch={lab.currentEpoch}
            currentLoss={lab.currentLoss}
            currentAccuracy={lab.currentAccuracy}
            logs={lab.logs}
            onNavigate={setActivePage}
            onStartTraining={lab.startTraining}
            onResetModel={lab.resetModel}
            onTriggerPulse={lab.triggerRandomPulse}
          />
        );
      case 'network':
        return (
          <NetworkBuilderPage
            architecture={lab.architecture}
            snapshot={lab.snapshot}
            taskType={lab.dataset.meta.taskType}
            isTraining={lab.trainingStatus === 'training'}
            onAddHiddenLayer={lab.addHiddenLayer}
            onRemoveHiddenLayer={lab.removeHiddenLayer}
            onUpdateHiddenLayer={lab.updateHiddenLayer}
            onUpdateOutputActivation={lab.updateOutputActivation}
            onApplyPreset={lab.applyArchitecturePreset}
            onRebuildModel={lab.resetModel}
            onTriggerPulse={lab.triggerRandomPulse}
          />
        );
      case 'datasets':
        return (
          <DatasetsPage
            dataset={lab.dataset}
            predictionVisuals={lab.predictionVisuals}
            isTraining={lab.trainingStatus === 'training'}
            onSelectDataset={lab.selectDataset}
            onUpdateDatasetConfig={lab.updateDatasetConfig}
            onRegenerate={lab.regenerateCurrentDataset}
            onProbeCoordinate={lab.probeNetwork}
            onNavigate={setActivePage}
          />
        );
      case 'training':
        return (
          <TrainingPage
            architecture={lab.architecture}
            snapshot={lab.snapshot}
            dataset={lab.dataset}
            predictionVisuals={lab.predictionVisuals}
            trainingConfig={lab.trainingConfig}
            trainingStatus={lab.trainingStatus}
            currentEpoch={lab.currentEpoch}
            currentLoss={lab.currentLoss}
            currentAccuracy={lab.currentAccuracy}
            elapsedMs={lab.elapsedMs}
            onSelectDataset={lab.selectDataset}
            onUpdateTrainingConfig={lab.updateTrainingConfig}
            onStartTraining={lab.startTraining}
            onStopTraining={lab.stopTraining}
            onResetModel={lab.resetModel}
            onProbeCoordinate={lab.probeNetwork}
            onTriggerPulse={lab.triggerRandomPulse}
            onNavigate={setActivePage}
          />
        );
      case 'analytics':
        return (
          <AnalyticsPage
            liveHistory={lab.liveHistory}
            runsHistory={lab.runsHistory}
            trainingStatus={lab.trainingStatus}
            architecture={lab.architecture}
            dataset={lab.dataset}
            trainingConfig={lab.trainingConfig}
            onStartTraining={lab.startTraining}
            onNavigate={setActivePage}
          />
        );
    }
  };

  return (
    <div className={`crt-root ${crtEnabled ? '' : 'crt-disabled'}`}>
      <div className="crt-overlay-scanlines" aria-hidden="true" />

      <Header
        systemReady={lab.systemReady}
        tfBackend={lab.tfBackend}
        totalLayers={calculateTotalLayers(lab.architecture)}
        totalParams={lab.modelParamsCount}
        trainingStatus={lab.trainingStatus}
        currentEpoch={lab.currentEpoch}
        targetEpochs={lab.trainingConfig.epochs}
        currentLoss={lab.currentLoss}
        currentAccuracy={lab.currentAccuracy}
        crtEnabled={crtEnabled}
        onToggleCrt={() => setCrtEnabled((v) => !v)}
        onStartTraining={lab.startTraining}
        onStopTraining={lab.stopTraining}
      />

      <div className="app-body">
        <Sidebar
          activePage={activePage}
          onNavigate={setActivePage}
          tfBackend={lab.tfBackend}
          tensorCount={lab.tensorCount}
          datasetCode={lab.dataset.meta.code}
          terminalOpen={terminalOpen}
          onToggleTerminal={() => setTerminalOpen((v) => !v)}
          logCount={lab.logs.length}
        />

        <div className="app-workspace">
          {!lab.systemReady ? (
            <div className="app-page-scroll">
              <div className="tech-panel" style={{ padding: '28px 24px', textAlign: 'center' }}>
                <div className="phosphor-text" style={{ fontSize: 15, fontWeight: 700 }}>
                  ИНИЦИАЛИЗАЦИЯ ВЫЧИСЛИТЕЛЬНОГО КОМПЛЕКСА TENSORFLOW.JS
                  <span className="blinking-cursor" />
                </div>
                <p style={{ color: 'var(--text-muted)', marginTop: 8 }}>
                  Подключение аппаратного ускорителя и выделение тензорных буферов...
                </p>
              </div>
            </div>
          ) : (
            <main className="app-page-scroll">{renderActivePage()}</main>
          )}

          <Terminal
            logs={lab.logs}
            isOpen={terminalOpen}
            onToggle={() => setTerminalOpen((v) => !v)}
            onClear={lab.clearLogs}
            onExecuteCommand={lab.executeCommand}
          />
        </div>
      </div>
    </div>
  );
};

export default App;
