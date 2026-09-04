'use client';

import { useState } from 'react';
import { Header } from '@/components/layout/Header';
import { MetricCard } from '@/components/common/MetricCard';
import { StatusBadge } from '@/components/common/StatusBadge';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { portfolioApi, riskApi, tradesApi, marketsApi } from '@/lib/api';
import { clsx } from 'clsx';
import { AlertTriangle, Zap } from 'lucide-react';
import type { Position, Trade } from '@/types';

function formatINR(v: number) {
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(v);
}

function SquareOffModal({ onClose, onConfirm, isPending }: { onClose: () => void; onConfirm: () => void; isPending: boolean }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-surface border border-border rounded-xl p-6 w-full max-w-sm mx-4 animate-fade-in">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-full bg-loss/20 flex items-center justify-center">
            <AlertTriangle className="w-5 h-5 text-loss" />
          </div>
          <div>
            <h3 className="font-semibold text-text-primary">Square Off All Intraday</h3>
            <p className="text-xs text-muted">This will close all open intraday positions</p>
          </div>
        </div>
        <div className="bg-loss/10 border border-loss/20 rounded p-3 mb-4 text-xs text-text-secondary">
          Market orders will be placed to close all positions immediately. This cannot be undone.
        </div>
        <div className="flex gap-2">
          <button onClick={onClose} className="flex-1 px-4 py-2 border border-border rounded text-sm text-text-secondary hover:border-muted transition-colors">
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={isPending}
            className="flex-1 px-4 py-2 bg-loss text-white rounded text-sm font-semibold hover:bg-red-600 disabled:opacity-50 transition-colors"
          >
            {isPending ? 'Squaring Off...' : 'Square Off All'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default function IntradayPage() {
  const [showSquareOff, setShowSquareOff] = useState(false);
  const queryClient = useQueryClient();

  const { data: positions } = useQuery({
    queryKey: ['portfolio', 'positions', 'intraday'],
    queryFn: () => portfolioApi.getPositions('intraday'),
    refetchInterval: 10_000,
  });

  const { data: riskState } = useQuery({
    queryKey: ['risk', 'state'],
    queryFn: () => riskApi.getState(),
    refetchInterval: 15_000,
  });

  const { data: trades } = useQuery({
    queryKey: ['trades', 'intraday-today'],
    queryFn: () => tradesApi.getTrades({ strategy: 'intraday' }),
    refetchInterval: 30_000,
  });

  const { data: regime } = useQuery({
    queryKey: ['markets', 'regime'],
    queryFn: () => marketsApi.getRegime(),
    refetchInterval: 60_000,
  });

  const squareOff = useMutation({
    mutationFn: () => riskApi.squareOffAll('intraday'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolio'] });
      setShowSquareOff(false);
    },
  });

  const intradayPositions = positions ?? [];
  const todayTrades = trades?.trades ?? [];

  // Compute today summary
  const grossPnl = todayTrades.reduce((s, t) => s + t.grossPnl, 0);
  const netPnl = todayTrades.reduce((s, t) => s + t.netPnl, 0);
  const costs = todayTrades.reduce((s, t) => s + t.costs, 0);
  const wins = todayTrades.filter((t) => t.outcome === 'WIN').length;
  const winRate = todayTrades.length > 0 ? (wins / todayTrades.length) * 100 : 0;

  // From the backend's limit table rather than from fields it never sent.
  // `?? 0` against `?? 5000` drew a comfortable empty bar whenever the daily
  // loss was not measured — which was always.
  const dailyLimit = riskState?.limits?.find((l) => l.name === 'max_daily_loss_pct');
  const dailyLoss = dailyLimit?.current != null ? Math.abs(dailyLimit.current) : null;
  const maxDailyLoss = dailyLimit ? Math.abs(dailyLimit.limit) : null;
  const dailyUsedPct =
    dailyLoss != null && maxDailyLoss ? (dailyLoss / maxDailyLoss) * 100 : null;

  return (
    <div className="flex flex-col h-full">
      <Header title="Intraday" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Today Summary */}
        <section>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <MetricCard title="Trades Today" value={todayTrades.length} format="number" />
            <MetricCard title="Gross P&L" value={grossPnl} format="currency" />
            <MetricCard title="Net P&L" value={netPnl} format="currency" />
            <MetricCard title="Transaction Costs" value={costs} format="currency" invertColors />
            <MetricCard title="Win Rate" value={winRate} format="percent" subtitle={`${wins}/${todayTrades.length} trades`} />
          </div>
        </section>

        {/* Daily Risk Meter + Regime + Square Off */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Daily Risk */}
          <div className="bg-surface border border-border rounded-lg p-4 col-span-2">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-text-primary">Daily Risk Meter</h3>
              <span className={clsx('text-xs font-mono', dailyUsedPct == null ? 'text-muted' : dailyUsedPct >= 85 ? 'text-loss' : dailyUsedPct >= 60 ? 'text-warning' : 'text-muted')}>
                {dailyLoss != null && maxDailyLoss != null
                  ? `${formatINR(dailyLoss)} / ${formatINR(maxDailyLoss)}`
                  : 'not measured'}
              </span>
            </div>
            <div className="progress-bar mb-2">
              {/* No bar at all when the loss is not measured. A zero-width bar
                  against a limit reads as "well within budget", which is a
                  claim nothing has established. */}
              <div
                className={clsx('progress-bar-fill h-3', dailyUsedPct == null ? 'bg-border' : dailyUsedPct >= 85 ? 'bg-loss' : dailyUsedPct >= 60 ? 'bg-warning' : 'bg-accent')}
                style={{ width: dailyUsedPct == null ? '0%' : `${Math.min(dailyUsedPct, 100)}%` }}
              />
            </div>
            <p className="text-xs text-muted">
              {dailyUsedPct != null
                ? `${dailyUsedPct.toFixed(1)}% of daily loss limit used`
                : 'Daily loss is not measured: it needs an intraday P&L series, which is not yet persisted.'}
            </p>
          </div>

          {/* Market Regime */}
          <div className="bg-surface border border-border rounded-lg p-4">
            <h3 className="text-sm font-semibold text-text-primary mb-3">Market Regime</h3>
            {regime ? (
              <div className="space-y-2">
                <StatusBadge status={regime.trend === 'UP' ? 'HEALTHY' : regime.trend === 'DOWN' ? 'DISABLED' : 'REDUCED'} size="md" />
                <p className="text-xs text-text-secondary">{regime.regime.replace(/_/g, ' ')}</p>
                <p className="text-xs text-muted">Confidence: {(regime.confidence * 100).toFixed(0)}%</p>
                <p className="text-xs text-muted">VIX: {regime.vix.toFixed(1)}</p>
              </div>
            ) : (
              <p className="text-xs text-muted">Loading...</p>
            )}
          </div>
        </section>

        {/* Open Intraday Positions */}
        <section className="bg-surface border border-border rounded-lg">
          <div className="flex items-center justify-between px-4 py-3 border-b border-border">
            <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
              <Zap className="w-4 h-4 text-accent" />
              Open Positions ({intradayPositions.length})
            </h3>
            <button
              onClick={() => setShowSquareOff(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-loss/20 border border-loss/30 text-loss rounded text-xs font-semibold hover:bg-loss/30 transition-colors"
            >
              <AlertTriangle className="w-3.5 h-3.5" />
              Square Off All
            </button>
          </div>

          {intradayPositions.length === 0 ? (
            <div className="py-10 text-center text-muted text-sm">No open intraday positions</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Qty</th>
                    <th>Avg Price</th>
                    <th>CMP</th>
                    <th>P&L</th>
                    <th>P&L %</th>
                    <th>Stop</th>
                    <th>Target</th>
                  </tr>
                </thead>
                <tbody>
                  {intradayPositions.map((pos: Position) => (
                    <tr key={pos.id}>
                      <td className="font-semibold">{pos.symbol}</td>
                      <td className="font-mono tabular-nums">{pos.qty}</td>
                      <td className="font-mono tabular-nums">{formatINR(pos.avgPrice)}</td>
                      <td className="font-mono tabular-nums">{formatINR(pos.cmp)}</td>
                      <td className={clsx('font-mono tabular-nums', pos.unrealizedPnl >= 0 ? 'text-profit' : 'text-loss')}>
                        {pos.unrealizedPnl >= 0 ? '+' : ''}{formatINR(pos.unrealizedPnl)}
                      </td>
                      <td className={clsx('font-mono tabular-nums', pos.unrealizedPnlPct >= 0 ? 'text-profit' : 'text-loss')}>
                        {pos.unrealizedPnlPct >= 0 ? '+' : ''}{pos.unrealizedPnlPct.toFixed(2)}%
                      </td>
                      <td className="font-mono tabular-nums text-loss">{formatINR(pos.stopLoss)}</td>
                      <td className="font-mono tabular-nums text-profit">{formatINR(pos.target)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* Today's Trades */}
        <section className="bg-surface border border-border rounded-lg">
          <div className="px-4 py-3 border-b border-border">
            <h3 className="text-sm font-semibold text-text-primary">Today&apos;s Trades ({todayTrades.length})</h3>
          </div>

          {todayTrades.length === 0 ? (
            <div className="py-10 text-center text-muted text-sm">No trades executed today</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Direction</th>
                    <th>Qty</th>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>P&L</th>
                    <th>Net P&L</th>
                    <th>Slippage</th>
                    <th>Exit Reason</th>
                    <th>Outcome</th>
                  </tr>
                </thead>
                <tbody>
                  {todayTrades.map((trade: Trade) => (
                    <tr key={trade.id}>
                      <td className="font-semibold">{trade.symbol}</td>
                      <td>
                        <span className={clsx('text-xs font-semibold', trade.direction === 'LONG' ? 'text-profit' : 'text-loss')}>
                          {trade.direction}
                        </span>
                      </td>
                      <td className="font-mono tabular-nums">{trade.qty}</td>
                      <td className="font-mono tabular-nums">{formatINR(trade.entryPrice)}</td>
                      <td className="font-mono tabular-nums">{formatINR(trade.exitPrice)}</td>
                      <td className={clsx('font-mono tabular-nums', trade.grossPnl >= 0 ? 'text-profit' : 'text-loss')}>
                        {trade.grossPnl >= 0 ? '+' : ''}{formatINR(trade.grossPnl)}
                      </td>
                      <td className={clsx('font-mono tabular-nums', trade.netPnl >= 0 ? 'text-profit' : 'text-loss')}>
                        {trade.netPnl >= 0 ? '+' : ''}{formatINR(trade.netPnl)}
                      </td>
                      <td className="font-mono tabular-nums text-muted">{formatINR(trade.slippage)}</td>
                      <td className="text-xs text-text-secondary">{trade.exitReason}</td>
                      <td>
                        <StatusBadge
                          status={trade.outcome === 'WIN' ? 'HEALTHY' : trade.outcome === 'LOSS' ? 'DISABLED' : 'REDUCED'}
                          size="sm"
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>

      {showSquareOff && (
        <SquareOffModal
          onClose={() => setShowSquareOff(false)}
          onConfirm={() => squareOff.mutate()}
          isPending={squareOff.isPending}
        />
      )}
    </div>
  );
}
