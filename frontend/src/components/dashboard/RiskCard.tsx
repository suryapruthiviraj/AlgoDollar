'use client';

import { useQuery } from '@tanstack/react-query';
import { riskApi, strategiesApi } from '@/lib/api';
import { StatusBadge } from '@/components/common/StatusBadge';
import { clsx } from 'clsx';
import { AlertTriangle, Shield } from 'lucide-react';
import type { StrategyName } from '@/types';

const REGIME_LABELS: Record<string, string> = {
  BULL_TRENDING: 'Bull Trending',
  BEAR_TRENDING: 'Bear Trending',
  BULL_VOLATILE: 'Bull Volatile',
  BEAR_VOLATILE: 'Bear Volatile',
  SIDEWAYS_LOW_VOL: 'Sideways (Low Vol)',
  SIDEWAYS_HIGH_VOL: 'Sideways (High Vol)',
};

const STRATEGY_NAMES: StrategyName[] = ['intraday', 'swing', 'longterm'];
const STRATEGY_LABELS: Record<StrategyName, string> = {
  intraday: 'Intraday',
  swing: 'Swing',
  longterm: 'Long-Term',
};

interface ProgressBarProps {
  used: number;
  max: number;
  label: string;
  format?: 'currency' | 'percent';
  warningAt?: number;
  dangerAt?: number;
}

function ProgressBar({ used, max, label, format = 'percent', warningAt = 0.7, dangerAt = 0.9 }: ProgressBarProps) {
  const ratio = max > 0 ? Math.min(used / max, 1) : 0;
  const pct = ratio * 100;

  const barColor =
    ratio >= dangerAt
      ? 'bg-loss'
      : ratio >= warningAt
      ? 'bg-warning'
      : 'bg-accent';

  const formatVal = (v: number) =>
    format === 'currency'
      ? `₹${new Intl.NumberFormat('en-IN', { notation: 'compact' }).format(Math.abs(v))}`
      : `${Math.abs(v).toFixed(1)}%`;

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-muted">{label}</span>
        <span className={clsx('font-mono', ratio >= dangerAt ? 'text-loss' : ratio >= warningAt ? 'text-warning' : 'text-text-secondary')}>
          {formatVal(used)} / {formatVal(max)}
        </span>
      </div>
      <div className="progress-bar">
        <div
          className={clsx('progress-bar-fill', barColor)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function RiskCard() {
  const { data: riskState, isLoading: riskLoading } = useQuery({
    queryKey: ['risk', 'state'],
    queryFn: () => riskApi.getState(),
    refetchInterval: 15_000,
    retry: false,
  });

  const { data: strategies } = useQuery({
    queryKey: ['strategies'],
    queryFn: () => strategiesApi.getAll(),
    refetchInterval: 30_000,
    retry: false,
  });

  const activeBreaches = riskState?.activeBreach ?? [];
  const hasBreaches = activeBreaches.length > 0;

  return (
    <div className="bg-surface border border-border rounded-lg p-4 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
          <Shield className="w-4 h-4 text-accent" />
          Risk Dashboard
        </h3>
        {riskState?.killSwitchActive && (
          <span className="text-xs font-semibold text-loss bg-loss/15 border border-loss/30 px-2 py-0.5 rounded">
            KILL SWITCH ON
          </span>
        )}
      </div>

      {riskLoading ? (
        <div className="space-y-3 animate-pulse">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-8 bg-border rounded" />
          ))}
        </div>
      ) : (
        <>
          {/* Risk Limit Breaches */}
          {hasBreaches && (
            <div className="space-y-1.5">
              {activeBreaches.map((breach, i) => (
                <div
                  key={i}
                  className={clsx(
                    'flex items-start gap-2 p-2.5 rounded text-xs border',
                    breach.severity === 'CRITICAL'
                      ? 'bg-loss/15 border-loss/30 text-loss'
                      : breach.severity === 'HIGH'
                      ? 'bg-loss/10 border-loss/20 text-loss'
                      : 'bg-warning/10 border-warning/20 text-warning',
                  )}
                >
                  <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                  <span>{breach.message}</span>
                </div>
              ))}
            </div>
          )}

          {/* Progress bars */}
          <div className="space-y-3">
            <ProgressBar
              label="Portfolio Drawdown"
              used={Math.abs(riskState?.portfolioDrawdown ?? 0)}
              max={Math.abs(riskState?.maxAllowedDrawdown ?? 15)}
              format="percent"
              warningAt={0.7}
              dangerAt={0.9}
            />
            <ProgressBar
              label="Daily Loss Limit"
              used={Math.abs(riskState?.dailyLoss ?? 0)}
              max={Math.abs(riskState?.maxDailyLoss ?? 5000)}
              format="currency"
              warningAt={0.6}
              dangerAt={0.85}
            />
            <ProgressBar
              label="Monthly Loss Limit"
              used={Math.abs(riskState?.monthlyLoss ?? 0)}
              max={Math.abs(riskState?.maxMonthlyLoss ?? 20000)}
              format="currency"
              warningAt={0.6}
              dangerAt={0.85}
            />
          </div>

          {/* Strategy Health */}
          <div>
            <p className="text-xs text-muted mb-2">Strategy Health</p>
            <div className="flex flex-col gap-1.5">
              {STRATEGY_NAMES.map((name) => {
                const health = strategies?.find((s) => s.strategy === name);
                const status = health?.status ?? 'UNKNOWN';
                return (
                  <div key={name} className="flex items-center justify-between">
                    <span className="text-xs text-text-secondary">{STRATEGY_LABELS[name]}</span>
                    <StatusBadge status={status} size="sm" />
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
