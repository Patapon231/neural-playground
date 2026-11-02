import React, { useEffect, useRef, useState } from 'react';
import { ChevronDown, ChevronUp, Trash2, TerminalSquare } from 'lucide-react';
import { TerminalLogEntry } from '../../types';
import './Terminal.css';

interface TerminalProps {
  logs: TerminalLogEntry[];
  isOpen: boolean;
  onToggle: () => void;
  onClear: () => void;
  onExecuteCommand: (rawCommand: string) => void;
}

export const Terminal: React.FC<TerminalProps> = ({
  logs,
  isOpen,
  onToggle,
  onClear,
  onExecuteCommand,
}) => {
  const [input, setInput] = useState('');
  const [cmdHistory, setCmdHistory] = useState<string[]>([]);
  const [historyIdx, setHistoryIdx] = useState<number>(-1);
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (isOpen && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [logs, isOpen]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed) return;
    setCmdHistory((prev) => [...prev, trimmed]);
    setHistoryIdx(-1);
    setInput('');
    onExecuteCommand(trimmed);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (cmdHistory.length === 0) return;
      const nextIdx = historyIdx === -1 ? cmdHistory.length - 1 : Math.max(0, historyIdx - 1);
      setHistoryIdx(nextIdx);
      setInput(cmdHistory[nextIdx]);
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (historyIdx === -1) return;
      if (historyIdx >= cmdHistory.length - 1) {
        setHistoryIdx(-1);
        setInput('');
      } else {
        const nextIdx = historyIdx + 1;
        setHistoryIdx(nextIdx);
        setInput(cmdHistory[nextIdx]);
      }
    }
  };

  const latestLog = logs[logs.length - 1];

  return (
    <section
      className={`np-terminal ${isOpen ? 'is-open' : 'is-collapsed'}`}
      aria-label="Системный терминал"
    >
      <div className="np-terminal-bar" onClick={onToggle}>
        <div className="np-terminal-title">
          <TerminalSquare size={14} />
          <span className="np-terminal-code">NP-OS // КОНСОЛЬ СИСТЕМЫ</span>
          <span className="np-terminal-hint">
            [команды: help, status, info, train, stop, reset, clear]
          </span>
        </div>

        {!isOpen && latestLog && (
          <div className="np-terminal-preview">
            <span className="np-log-time">[{latestLog.timestamp}]</span>{' '}
            <span className="np-log-cat">[{latestLog.category}]</span> {latestLog.message}
          </div>
        )}

        <div className="np-terminal-controls" onClick={(e) => e.stopPropagation()}>
          <button
            type="button"
            className="retro-btn retro-btn-sm"
            onClick={onClear}
            title="Очистить журнал терминала"
          >
            <Trash2 size={12} />
            ОЧИСТИТЬ
          </button>
          <button
            type="button"
            className="retro-btn retro-btn-sm"
            onClick={onToggle}
            title={isOpen ? 'Свернуть терминал' : 'Развернуть терминал'}
          >
            {isOpen ? <ChevronDown size={13} /> : <ChevronUp size={13} />}
            {isOpen ? 'СВЕРНУТЬ' : 'ОТКРЫТЬ'}
          </button>
        </div>
      </div>

      {isOpen && (
        <>
          <div
            className="np-terminal-body"
            ref={bodyRef}
            onClick={() => inputRef.current?.focus()}
          >
            {logs.length === 0 ? (
              <div className="np-terminal-empty">
                Журнал очищен. Введите &quot;help&quot; для вывода списка команд.
              </div>
            ) : (
              logs.map((entry) => (
                <div
                  key={entry.id}
                  className={`np-log-line ${entry.isError ? 'is-error' : ''} cat-${entry.category}`}
                >
                  <span className="np-log-time">[{entry.timestamp}]</span>
                  <span className="np-log-cat">[{entry.category}]</span>
                  <span className="np-log-msg">{entry.message}</span>
                </div>
              ))
            )}
          </div>

          <form className="np-terminal-prompt" onSubmit={handleSubmit}>
            <span className="np-prompt-prefix">NP-88&gt;</span>
            <input
              ref={inputRef}
              type="text"
              className="np-prompt-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Введите команду (например: help, status, info, clear)..."
              spellCheck={false}
              autoComplete="off"
            />
            <button type="submit" className="retro-btn retro-btn-sm">
              ВВОД ↵
            </button>
          </form>
        </>
      )}
    </section>
  );
};
