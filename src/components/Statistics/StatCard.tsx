import React from 'react';
import './Statistics.css';

interface StatCardProps {
  code: string;
  label: string;
  value: string | number;
  subtext?: string;
  accent?: boolean;
  icon?: React.ReactNode;
}

export const StatCard: React.FC<StatCardProps> = ({
  code,
  label,
  value,
  subtext,
  accent = false,
  icon,
}) => {
  return (
    <div className={`tech-panel np-stat-card ${accent ? 'is-accent' : ''}`}>
      <div className="np-stat-top">
        <span className="np-stat-code">[{code}]</span>
        <span className="np-stat-label">{label}</span>
        {icon && <span className="np-stat-icon">{icon}</span>}
      </div>
      <div className={`np-stat-value ${accent ? 'phosphor-text' : ''}`}>{value}</div>
      {subtext && <div className="np-stat-subtext">{subtext}</div>}
    </div>
  );
};
