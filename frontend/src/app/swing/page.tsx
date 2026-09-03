'use client';

import { Header } from '@/components/layout/Header';
import { MetricCard } from '@/components/common/MetricCard';
import { StatusBadge } from '@/components/common/StatusBadge';
import { useQuery } from '@tanstack/react-query';
import { portfolioApi, strategiesApi } from '@/lib/api';
import { clsx } from 'clsx';
import { TrendingUp } from 'lucide-react';
import type { Position, Signal } from '@/types';

function formatINR(v: number) {
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(v);
}

export default function SwingPage() {
  const { data: positions } = useQuery({
    queryKey: ['portfolio', 'positions', 'swing'],
    queryFn: () => portfolioApi.getPositions('swing'),
    refetchInterval: 30_000,
  });

  const { data: signals } = useQuery({
    queryKey: ['strategies', 'swing', 'signals'],
    queryFn: () => strategiesApi.getSignals('swing'),
    refetchInterval: 60_000,
  });

  const { data: performance } = useQuery({
    queryKey: ['strategies', 'swing', 'performance'],
    queryFn: () => strategiesApi.getPerformance('swing'),
    staleTime: 60_000,
  });

  const swingPositions = positions ?? [];
  const pendingSignals = (signals ?? []).filter((s: Signal) => s.status === 'PENDING');

  return (
    <div className="flex flex-col h-full">
      <Header title="Swing Trading" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Performance Stats */}
        <section className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          <MetricCard
            title="30D Return"
            value={performance?.return30d ?? null}
            format="percent"
            loading={!performance}
          />
          <MetricCard
            title="Win Rate"
            value={performance?.winRate ?? null}
            format="percent"
            loading={!performance}
          />
          <MetricCard
            title="Sharpe Ratio"
            value={performance?.sharpe ?? null}
            format="number"
            loading={!performance}
          />
          <MetricCard
            title="Avg Win"
            value={performance?.avgWin ?? null}
            format="percent"
            loading={!performance}
          />
          <MetricCard
            title="Avg Loss"
            value={performance?.avgLoss ?? null}
            format="percent"
            invertColors
            loading={!performance}
          />
          <MetricCard
            title="Profit Factor"
            value={performance?.profitFactor ?? null}
            format="number"
            loading={!performance}
          />
        </section>

        {/* Current Swing Positions */}
        <section className="bg-surface border border-border rounded-lg">
          <div className="flex items-center gap-2 px-4 py-3 border-b border-border">
            <TrendingUp className="w-4 h-4 text-accent" />
            <h3 className="text-sm font-semibold text-text-primary">
              Current Swing Positions ({swingPositions.length})
            </h3>
          </div>

          {swingPositions.length === 0 ? (
            <div className="py-10 text-center text-muted text-sm">No open swing positions</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Entry</th>
                    <th>CMP</th>
                    <th>Stop</th>
                    <th>Target</th>
                    <th>Days Held</th>
                    <th>P&L</th>
                    <th>P&L %</th>
                    <th>Weight</th>
                  </tr>
                </thead>
                <tbody>
                  {swingPositions.map((pos: Position) => (
                    <tr key={pos.id}>
                      <td>
                        <div className="font-semibold text-text-primary">{pos.symbol}</div>
                        <div className="text-2xs text-muted">{pos.sector}</div>
                      </td>
                      <td className="font-mono tabular-nums">{formatINR(pos.avgPrice)}</td>
                      <td className="font-mono tabular-nums">{formatINR(pos.cmp)}</td>
                      <td className="font-mono tabular-nums text-loss">{formatINR(pos.stopLoss)}</td>
                      <td className="font-mono tabular-nums text-profit">{formatINR(pos.target)}</td>
                      <td className="font-mono tabular-nums text-text-secondary">{pos.daysHeld}d</td>
                      <td className={clsx('font-mono tabular-nums', pos.unrealizedPnl >= 0 ? 'text-profit' : 'text-loss')}>
                        {pos.unrealizedPnl >= 0 ? '+' : ''}{formatINR(pos.unrealizedPnl)}
                      </td>
                      <td className={clsx('font-mono tabular-nums', pos.unrealizedPnlPct >= 0 ? 'text-profit' : 'text-loss')}>
                        {pos.unrealizedPnlPct >= 0 ? '+' : ''}{pos.unrealizedPnlPct.toFixed(2)}%
                      </td>
                      <td className="font-mono tabular-nums text-muted">{pos.portfolioWeight.toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* Signal Pipeline */}
        <section className="bg-surface border border-border rounded-lg">
          <div className="px-4 py-3 border-b border-border">
            <h3 className="text-sm font-semibold text-text-primary">
              Signal Pipeline ({pendingSignals.length} pending)
            </h3>
            <p className="text-xs text-muted mt-0.5">Ranked opportunities not yet entered</p>
          </div>

          {pendingSignals.length === 0 ? (
            <div className="py-10 text-center text-muted text-sm">No pending signals</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Direction</th>
                    <th>Score</th>
                    <th>Strength</th>
                    <th>Entry</th>
                    <th>Stop</th>
                    <th>Target</th>
                    <th>Exp. Return</th>
                    <th>Hold (days)</th>
                    <th>Generated</th>
                  </tr>
                </thead>
                <tbody>
                  {pendingSignals.map((sig: Signal, i: number) => (
                    <tr key={sig.id}>
                      <td>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-muted font-mono">#{i + 1}</span>
                          <span className="font-semibold text-text-primary">{sig.symbol}</span>
                        </div>
                      </td>
                      <td>
                        <span className={clsx('text-xs font-semibold', sig.direction === 'LONG' ? 'text-profit' : 'text-loss')}>
                          {sig.direction}
                        </span>
                      </td>
                      <td className="font-mono tabular-nums">
                        <div className="flex items-center gap-2">
                          <div className="w-12 h-1.5 bg-border rounded-full overflow-hidden">
                            <div className="h-full bg-accent" style={{ width: `${sig.score * 100}%` }} />
                          </div>
                          <span className="text-xs text-text-secondary">{(sig.score * 100).toFixed(0)}</span>
                        </div>
                      </td>
                      <td>
                        <StatusBadge
                          status={sig.strength === 'STRONG' ? 'HEALTHY' : sig.strength === 'MODERATE' ? 'REDUCED' : 'PAUSED'}
                          size="sm"
                        />
                      </td>
                      <td className="font-mono tabular-nums">{formatINR(sig.entryPrice)}</td>
                      <td className="font-mono tabular-nums text-loss">{formatINR(sig.stopLoss)}</td>
                      <td className="font-mono tabular-nums text-profit">{formatINR(sig.target)}</td>
                      <td className={clsx('font-mono tabular-nums', sig.expectedReturn >= 0 ? 'text-profit' : 'text-loss')}>
                        {sig.expectedReturn >= 0 ? '+' : ''}{sig.expectedReturn.toFixed(2)}%
                      </td>
                      <td className="font-mono tabular-nums text-muted">{sig.expectedHoldDays}d</td>
                      <td className="text-xs text-muted">
                        {new Date(sig.generatedAt).toLocaleDateString('en-IN')}
                      </td>
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
