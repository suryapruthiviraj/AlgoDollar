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

  // Plain strings from the backend: each names one condition currently
  // blocking trading. They carried a `severity` field the API never sent.
  const activeBreaches = riskState?.activeBreaches ?? [];
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
                  className="flex items-start gap-2 p-2.5 rounded text-xs border bg-loss/15 border-loss/30 text-loss"
                >
                  <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                  <span>{breach}</span>
                </div>
              ))}
            </div>
          )}

          {/* Limits, from the backend's own table.
              The three bars here previously read fields the API never sent
              (`dailyLoss`, `maxAllowedDrawdown`, `monthlyLoss`) with invented
              fallbacks — `?? 0` used against `?? 15`, which drew an empty,
              reassuring bar whenever nothing was measured. A limit with no
              current reading now says so instead. */}
          <div className="space-y-3">
            {(riskState?.limits ?? []).slice(0, 4).map((l) =>
              l.measurable && l.current != null ? (
                <ProgressBar
                  key={l.name}
                  label={l.label}
                  used={Math.abs(l.current)}
                  max={Math.abs(l.limit)}
                  format={l.name.includes('pct') ? 'percent' : undefined}
                  warningAt={0.7}
                  dangerAt={0.9}
                />
              ) : (
                <div key={l.name} className="flex justify-between text-xs">
                  <span className="text-muted">{l.label}</span>
                  <span className="text-muted">not measured</span>
                </div>
              ),
            )}
            {!riskState?.limits?.length && (
              <p className="text-xs text-muted">Risk limits could not be read.</p>
            )}
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
