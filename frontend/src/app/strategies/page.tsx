'use client';

import { useState } from 'react';
import { Header } from '@/components/layout/Header';
import { StatusBadge } from '@/components/common/StatusBadge';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { strategiesApi } from '@/lib/api';
import { clsx } from 'clsx';
import { Bot, CheckCircle, X, Zap, TrendingUp, Building2 } from 'lucide-react';
import type { StrategyName, StrategyHealth, Signal } from '@/types';

function formatINR(v: number) {
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(v);
}

const STRATEGY_CONFIG: Record<StrategyName, { label: string; icon: React.ReactNode; color: string }> = {
  intraday: { label: 'Intraday', icon: <Zap className="w-5 h-5" />, color: '#8b5cf6' },
  swing: { label: 'Swing', icon: <TrendingUp className="w-5 h-5" />, color: '#3b82f6' },
  longterm: { label: 'Long-Term', icon: <Building2 className="w-5 h-5" />, color: '#22c55e' },
};

function OverrideModal({
  strategy,
  currentStatus,
  onClose,
  onConfirm,
}: {
  strategy: StrategyName;
  currentStatus: string;
  onClose: () => void;
  onConfirm: (status: string, reason: string) => void;
}) {
  const [newStatus, setNewStatus] = useState<string>('PAUSED');
  const [reason, setReason] = useState('');

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-surface border border-border rounded-xl p-6 w-full max-w-md mx-4 animate-fade-in">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-text-primary">Manual Status Override</h3>
          <button onClick={onClose} className="text-muted hover:text-text-primary">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="mb-4">
          <p className="text-sm text-text-secondary mb-1">Strategy: <span className="text-text-primary capitalize">{strategy}</span></p>
          <p className="text-sm text-text-secondary">Current Status: <StatusBadge status={currentStatus} /></p>
        </div>

        <div className="mb-3">
          <label className="block text-xs text-muted mb-1.5">New Status</label>
          <select
            value={newStatus}
            onChange={(e) => setNewStatus(e.target.value)}
            className="w-full bg-surface-2 border border-border rounded px-3 py-2 text-sm text-text-primary focus:border-accent focus:outline-none"
          >
            <option value="HEALTHY">HEALTHY</option>
            <option value="REDUCED">REDUCED</option>
            <option value="PAUSED">PAUSED</option>
            <option value="DISABLED">DISABLED</option>
          </select>
        </div>

        <div className="mb-4">
          <label className="block text-xs text-muted mb-1.5">Reason (required)</label>
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="Explain the reason for this override..."
            rows={3}
            className="w-full bg-surface-2 border border-border rounded px-3 py-2 text-sm text-text-primary focus:border-accent focus:outline-none resize-none"
          />
        </div>

        <div className="flex gap-2">
          <button onClick={onClose} className="flex-1 px-4 py-2 border border-border rounded text-sm text-text-secondary hover:border-muted transition-colors">
            Cancel
          </button>
          <button
            onClick={() => onConfirm(newStatus, reason)}
            disabled={!reason.trim()}
            className="flex-1 px-4 py-2 bg-accent text-white rounded text-sm font-semibold hover:bg-accent-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Apply Override
          </button>
        </div>
      </div>
    </div>
  );
}

function StrategyCard({ health }: { health: StrategyHealth }) {
  const [showOverride, setShowOverride] = useState(false);
  const queryClient = useQueryClient();
  const config = STRATEGY_CONFIG[health.strategy];

  const override = useMutation({
    mutationFn: ({ status, reason }: { status: string; reason: string }) =>
      strategiesApi.updateStatus(health.strategy, status, reason),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['strategies'] });
      setShowOverride(false);
    },
  });

  const { data: performance } = useQuery({
    queryKey: ['strategies', health.strategy, 'performance'],
    queryFn: () => strategiesApi.getPerformance(health.strategy),
    staleTime: 60_000,
  });

  return (
    <>
      <div className="bg-surface border border-border rounded-lg p-5">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ backgroundColor: `${config.color}20`, color: config.color }}>
              {config.icon}
            </div>
            <div>
              <p className="font-semibold text-text-primary">{config.label}</p>
              {health.isManualOverride && (
                <p className="text-2xs text-warning">Manual override active</p>
              )}
            </div>
          </div>
          <StatusBadge status={health.status} size="md" />
        </div>

        {/* Metrics grid */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div className="bg-surface-2 rounded p-2.5">
            <p className="text-2xs text-muted">30D Return</p>
            <p className={clsx('text-sm font-mono font-semibold tabular-nums', (performance?.return30d ?? 0) >= 0 ? 'text-profit' : 'text-loss')}>
              {(performance?.return30d ?? 0) >= 0 ? '+' : ''}{(performance?.return30d ?? 0).toFixed(2)}%
            </p>
          </div>
          <div className="bg-surface-2 rounded p-2.5">
            <p className="text-2xs text-muted">Sharpe</p>
            <p className="text-sm font-mono font-semibold tabular-nums text-text-primary">
              {(performance?.sharpe ?? 0).toFixed(2)}
            </p>
          </div>
          <div className="bg-surface-2 rounded p-2.5">
            <p className="text-2xs text-muted">Win Rate</p>
            <p className="text-sm font-mono font-semibold tabular-nums text-text-primary">
              {(performance?.winRate ?? 0).toFixed(1)}%
            </p>
          </div>
          <div className="bg-surface-2 rounded p-2.5">
            <p className="text-2xs text-muted">Active Positions</p>
            <p className="text-sm font-mono font-semibold tabular-nums text-text-primary">
              {health.activePositions} / {health.positionLimit}
            </p>
          </div>
        </div>

        {/* Capital */}
        <div className="mb-4">
          <div className="flex justify-between text-xs mb-1">
            <span className="text-muted">Capital</span>
            <span className="font-mono text-text-secondary">
              {formatINR(health.capitalAllocated)} / {formatINR(health.capitalLimit)}
            </span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-bar-fill bg-accent"
              style={{ width: `${Math.min((health.capitalAllocated / Math.max(health.capitalLimit, 1)) * 100, 100)}%` }}
            />
          </div>
        </div>

        {/* Last signal */}
        <div className="flex justify-between items-center text-xs text-muted mb-4">
          <span>Last signal</span>
          <span>{new Date(health.lastSignalAt).toLocaleString('en-IN')}</span>
        </div>

        {/* Override button */}
        <button
          onClick={() => setShowOverride(true)}
          className="w-full px-3 py-2 border border-border rounded text-xs text-text-secondary hover:text-text-primary hover:border-muted transition-colors"
        >
          Manual Status Override
        </button>
      </div>

      {showOverride && (
        <OverrideModal
          strategy={health.strategy}
          currentStatus={health.status}
          onClose={() => setShowOverride(false)}
          onConfirm={(status, reason) => override.mutate({ status, reason })}
        />
      )}
    </>
  );
}

export default function StrategiesPage() {
  const { data: strategies, isLoading } = useQuery({
    queryKey: ['strategies'],
    queryFn: () => strategiesApi.getAll(),
    refetchInterval: 30_000,
  });

  const { data: intradaySignals } = useQuery({
    queryKey: ['strategies', 'intraday', 'signals'],
    queryFn: () => strategiesApi.getSignals('intraday'),
    staleTime: 60_000,
  });

  const { data: swingSignals } = useQuery({
    queryKey: ['strategies', 'swing', 'signals'],
    queryFn: () => strategiesApi.getSignals('swing'),
    staleTime: 60_000,
  });

  const recentSignals = [...(intradaySignals ?? []), ...(swingSignals ?? [])]
    .sort((a, b) => new Date(b.generatedAt).getTime() - new Date(a.generatedAt).getTime())
    .slice(0, 10);

  return (
    <div className="flex flex-col h-full">
      <Header title="Strategies" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Strategy Cards */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <Bot className="w-4 h-4 text-accent" />
            <h3 className="text-sm font-semibold text-text-primary">Strategy Overview</h3>
          </div>
          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="bg-surface border border-border rounded-lg p-5 h-48 animate-pulse" />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {(strategies ?? []).map((health: StrategyHealth) => (
                <StrategyCard key={health.strategy} health={health} />
              ))}
            </div>
          )}
        </section>

        {/* Recent Signals */}
        <section className="bg-surface border border-border rounded-lg">
          <div className="px-4 py-3 border-b border-border">
            <h3 className="text-sm font-semibold text-text-primary">Recent Signals ({recentSignals.length})</h3>
          </div>
          {recentSignals.length === 0 ? (
            <div className="py-10 text-center text-muted text-sm">No recent signals</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Strategy</th>
                    <th>Direction</th>
                    <th>Score</th>
                    <th>Entry</th>
                    <th>Stop</th>
                    <th>Target</th>
                    <th>Exp. Return</th>
                    <th>Status</th>
                    <th>Generated</th>
                  </tr>
                </thead>
                <tbody>
                  {recentSignals.map((sig: Signal) => (
                    <tr key={sig.id}>
                      <td className="font-semibold">{sig.symbol}</td>
                      <td className="text-xs capitalize text-text-secondary">{sig.strategy}</td>
                      <td>
                        <span className={clsx('text-xs font-semibold', sig.direction === 'LONG' ? 'text-profit' : 'text-loss')}>
                          {sig.direction}
                        </span>
                      </td>
                      <td className="font-mono tabular-nums">{(sig.score * 100).toFixed(0)}</td>
                      <td className="font-mono tabular-nums">{formatINR(sig.entryPrice)}</td>
                      <td className="font-mono tabular-nums text-loss">{formatINR(sig.stopLoss)}</td>
                      <td className="font-mono tabular-nums text-profit">{formatINR(sig.target)}</td>
                      <td className={clsx('font-mono tabular-nums', sig.expectedReturn >= 0 ? 'text-profit' : 'text-loss')}>
                        {sig.expectedReturn >= 0 ? '+' : ''}{sig.expectedReturn.toFixed(2)}%
                      </td>
                      <td>
                        <StatusBadge
                          status={sig.status === 'PENDING' ? 'PENDING' : sig.status === 'ENTERED' ? 'COMPLETE' : 'CLOSED'}
                          size="sm"
                        />
                      </td>
                      <td className="text-xs text-muted">{new Date(sig.generatedAt).toLocaleString('en-IN')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
