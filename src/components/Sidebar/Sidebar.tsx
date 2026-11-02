import React from 'react';
import {
  LayoutDashboard,
  Network,
  Database,
  Cpu,
  LineChart,
  TerminalSquare,
} from 'lucide-react';
import { PageId } from '../../types';
import './Sidebar.css';

interface SidebarProps {
  activePage: PageId;
  onNavigate: (page: PageId) => void;
  tfBackend: string;
  tensorCount: number;
  datasetCode: string;
  terminalOpen: boolean;
  onToggleTerminal: () => void;
  logCount: number;
}

interface NavItem {
  id: PageId;
  index: string;
  label: string;
  sublabel: string;
  icon: React.ReactNode;
}

const NAV_ITEMS: NavItem[] = [
  {
    id: 'dashboard',
    index: '01',
    label: 'Главная',
    sublabel: 'Сводка и топология',
    icon: <LayoutDashboard size={16} />,
  },
  {
    id: 'network',
    index: '02',
    label: 'Конструктор нейросети',
    sublabel: 'Слои и активации',
    icon: <Network size={16} />,
  },
  {
    id: 'datasets',
    index: '03',
    label: 'Наборы данных',
    sublabel: 'Выборки и шум',
    icon: <Database size={16} />,
  },
  {
    id: 'training',
    index: '04',
    label: 'Обучение',
    sublabel: 'Гиперпараметры TF.js',
    icon: <Cpu size={16} />,
  },
  {
    id: 'analytics',
    index: '05',
    label: 'Аналитика',
    sublabel: 'Графики Loss / Acc',
    icon: <LineChart size={16} />,
  },
];

export const Sidebar: React.FC<SidebarProps> = ({
  activePage,
  onNavigate,
  tfBackend,
  tensorCount,
  datasetCode,
  terminalOpen,
  onToggleTerminal,
  logCount,
}) => {
  return (
    <aside className="np-sidebar" aria-label="Основная навигация">
      <div className="np-sidebar-section-header">
        <span>НАВИГАЦИЯ // МОДУЛИ</span>
        <span>[05]</span>
      </div>

      <nav className="np-nav-list">
        {NAV_ITEMS.map((item) => {
          const active = activePage === item.id;
          return (
            <button
              key={item.id}
              type="button"
              className={`np-nav-item ${active ? 'is-active' : ''}`}
              onClick={() => onNavigate(item.id)}
              aria-current={active ? 'page' : undefined}
            >
              <span className="np-nav-index">{item.index}</span>
              <span className="np-nav-icon">{item.icon}</span>
              <span className="np-nav-text">
                <span className="np-nav-label">{item.label}</span>
                <span className="np-nav-sublabel">{item.sublabel}</span>
              </span>
              {active && <span className="np-nav-arrow">◄</span>}
            </button>
          );
        })}
      </nav>

      <div className="np-sidebar-spacer" />

      <div className="np-sidebar-console-toggle">
        <button
          type="button"
          className={`retro-btn np-terminal-btn ${terminalOpen ? 'is-open' : ''}`}
          onClick={onToggleTerminal}
        >
          <TerminalSquare size={15} />
          <span>ТЕРМИНАЛ [{logCount}]</span>
        </button>
      </div>

      <div className="np-sidebar-diagnostics">
        <div className="np-diag-header">СИСТЕМНЫЙ МОНИТОР</div>
        <div className="np-diag-row">
          <span>ЯДРО ВЫЧИСЛЕНИЙ:</span>
          <span className="np-diag-val">TF.JS 4.X</span>
        </div>
        <div className="np-diag-row">
          <span>УСКОРИТЕЛЬ:</span>
          <span className="np-diag-val">{tfBackend.toUpperCase()}</span>
        </div>
        <div className="np-diag-row">
          <span>АКТИВНЫЕ ТЕНЗОРЫ:</span>
          <span className="np-diag-val">{tensorCount}</span>
        </div>
        <div className="np-diag-row">
          <span>БУФЕР ДАННЫХ:</span>
          <span className="np-diag-val">{datasetCode}</span>
        </div>
      </div>
    </aside>
  );
};
