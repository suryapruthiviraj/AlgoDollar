'use client';

import { useState } from 'react';
import { Header } from '@/components/layout/Header';
import { MetricCard } from '@/components/common/MetricCard';
import { StatusBadge } from '@/components/common/StatusBadge';
import { useQuery } from '@tanstack/react-query';
import { tradesApi, type TradeFilters } from '@/lib/api';
import { clsx } from 'clsx';
import { X, Download } from 'lucide-react';
import type { Trade, AuditEntry } from '@/types';

function formatINR(v: number) {
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(v);
}

function TradeDetailModal({ trade, onClose }: { trade: Trade; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm overflow-y-auto py-8">
      <div className="bg-surface border border-border rounded-xl p-6 w-full max-w-2xl mx-4 animate-fade-in">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="font-semibold text-text-primary">{trade.symbol} — Trade Detail</h3>
            <p className="text-xs text-muted">{trade.strategy} | {trade.direction}</p>
          </div>
          <button onClick={onClose} className="text-muted hover:text-text-primary">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          <div className="bg-surface-2 rounded p-2.5">
            <p className="text-2xs text-muted">Entry</p>
            <p className="text-sm font-mono font-semibold tabular-nums">{formatINR(trade.entryPrice)}</p>
            <p className="text-2xs text-muted">{new Date(trade.entryDate).toLocaleDateString('en-IN')}</p>
          </div>
          <div className="bg-surface-2 rounded p-2.5">
            <p className="text-2xs text-muted">Exit</p>
            <p className="text-sm font-mono font-semibold tabular-nums">{formatINR(trade.exitPrice)}</p>
            <p className="text-2xs text-muted">{new Date(trade.exitDate).toLocaleDateString('en-IN')}</p>
          </div>
          <div className="bg-surface-2 rounded p-2.5">
            <p className="text-2xs text-muted">Gross P&L</p>
            <p className={clsx('text-sm font-mono font-semibold tabular-nums', trade.grossPnl >= 0 ? 'text-profit' : 'text-loss')}>
              {trade.grossPnl >= 0 ? '+' : ''}{formatINR(trade.grossPnl)}
            </p>
          </div>
          <div className="bg-surface-2 rounded p-2.5">
            <p className="text-2xs text-muted">Net P&L</p>
            <p className={clsx('text-sm font-mono font-semibold tabular-nums', trade.netPnl >= 0 ? 'text-profit' : 'text-loss')}>
              {trade.netPnl >= 0 ? '+' : ''}{formatINR(trade.netPnl)}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-3 mb-4">
          <div className="bg-surface-2 rounded p-2.5">
            <p className="text-2xs text-muted">Costs</p>
            <p className="text-sm font-mono tabular-nums text-text-secondary">{formatINR(trade.costs)}</p>
          </div>
          <div className="bg-surface-2 rounded p-2.5">
            <p className="text-2xs text-muted">Slippage</p>
            <p className="text-sm font-mono tabular-nums text-text-secondary">{formatINR(trade.slippage)}</p>
          </div>
          <div className="bg-surface-2 rounded p-2.5">
            <p className="text-2xs text-muted">Hold Period</p>
            <p className="text-sm font-mono tabular-nums text-text-secondary">{trade.holdPeriodDays}d</p>
          </div>
        </div>

        <div className="bg-surface-2 rounded p-2.5 mb-4">
          <p className="text-2xs text-muted mb-1">Exit Reason</p>
          <p className="text-sm text-text-secondary">{trade.exitReason}</p>
        </div>

        {/* Audit Trail */}
        {trade.auditTrail && trade.auditTrail.length > 0 && (
          <div>
            <p className="text-xs text-muted mb-2 font-semibold uppercase tracking-wide">Audit Trail</p>
            <div className="space-y-1.5 max-h-48 overflow-y-auto">
              {trade.auditTrail.map((entry: AuditEntry, i: number) => (
                <div key={i} className="flex gap-3 text-xs">
                  <span className="text-muted shrink-0 font-mono">
                    {new Date(entry.timestamp).toLocaleTimeString('en-IN')}
                  </span>
                  <span className="text-accent shrink-0">{entry.action}</span>
                  <span className="text-text-secondary">{entry.details}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function exportToCSV(trades: Trade[]) {
  const headers = ['Date', 'Symbol', 'Strategy', 'Direction', 'Qty', 'Entry', 'Exit', 'Gross P&L', 'Net P&L', 'Costs', 'Slippage', 'Hold Days', 'Exit Reason', 'Outcome'];
  const rows = trades.map((t) => [
    t.entryDate,
    t.symbol,
    t.strategy,
    t.direction,
    t.qty,
    t.entryPrice,
    t.exitPrice,
    t.grossPnl,
    t.netPnl,
    t.costs,
    t.slippage,
    t.holdPeriodDays,
    t.exitReason,
    t.outcome,
  ]);
  const csv = [headers, ...rows].map((r) => r.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'algodollar-trades.csv';
  a.click();
  URL.revokeObjectURL(url);
}

export default function TradesPage() {
  const [filters, setFilters] = useState<TradeFilters>({});
  const [selectedTrade, setSelectedTrade] = useState<Trade | null>(null);

  const { data: tradesData, isLoading } = useQuery({
    queryKey: ['trades', filters],
    queryFn: () => tradesApi.getTrades(filters),
    staleTime: 30_000,
  });

  const { data: summary } = useQuery({
    queryKey: ['trades', 'summary', filters],
    queryFn: () => tradesApi.getSummary(filters),
    staleTime: 30_000,
  });

  const trades = tradesData?.trades ?? [];
  const total = tradesData?.total ?? 0;

  return (
    <div className="flex flex-col h-full">
      <Header title="Trade Journal" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Summary Stats */}
        <section className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          <MetricCard title="Total Trades" value={summary?.totalTrades ?? total} format="number" loading={isLoading} />
          <MetricCard title="Win Rate" value={summary?.winRate ?? null} format="percent" loading={isLoading} />
          <MetricCard title="Gross P&L" value={summary?.grossPnl ?? null} format="currency" loading={isLoading} />
          <MetricCard title="Net P&L" value={summary?.netPnl ?? null} format="currency" loading={isLoading} />
          <MetricCard title="Total Costs" value={summary?.totalCosts ?? null} format="currency" invertColors loading={isLoading} />
        </section>

        {/* Filters */}
        <section className="bg-surface border border-border rounded-lg p-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
            <div>
              <label className="block text-2xs text-muted mb-1">From</label>
              <input
                type="date"
                value={filters.startDate ?? ''}
                onChange={(e) => setFilters((f) => ({ ...f, startDate: e.target.value || undefined }))}
                className="w-full bg-surface-2 border border-border rounded px-2 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-2xs text-muted mb-1">To</label>
              <input
                type="date"
                value={filters.endDate ?? ''}
                onChange={(e) => setFilters((f) => ({ ...f, endDate: e.target.value || undefined }))}
                className="w-full bg-surface-2 border border-border rounded px-2 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-2xs text-muted mb-1">Strategy</label>
              <select
                value={filters.strategy ?? ''}
                onChange={(e) => setFilters((f) => ({ ...f, strategy: (e.target.value as 'intraday' | 'swing' | 'longterm') || undefined }))}
                className="w-full bg-surface-2 border border-border rounded px-2 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
              >
                <option value="">All</option>
                <option value="intraday">Intraday</option>
                <option value="swing">Swing</option>
                <option value="longterm">Long-Term</option>
              </select>
            </div>
            <div>
              <label className="block text-2xs text-muted mb-1">Outcome</label>
              <select
                value={filters.outcome ?? ''}
                onChange={(e) => setFilters((f) => ({ ...f, outcome: (e.target.value as 'WIN' | 'LOSS' | 'BREAKEVEN') || undefined }))}
                className="w-full bg-surface-2 border border-border rounded px-2 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
              >
                <option value="">All</option>
                <option value="WIN">Win</option>
                <option value="LOSS">Loss</option>
                <option value="BREAKEVEN">Breakeven</option>
              </select>
            </div>
            <div className="flex items-end">
              <button
                onClick={() => setFilters({})}
                className="px-3 py-1.5 border border-border rounded text-xs text-muted hover:text-text-primary hover:border-muted transition-colors"
              >
                Clear Filters
              </button>
            </div>
            <div className="flex items-end">
              <button
                onClick={() => exportToCSV(trades)}
                disabled={trades.length === 0}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-surface-2 border border-border rounded text-xs text-text-secondary hover:text-text-primary disabled:opacity-50 transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                Export CSV
              </button>
            </div>
          </div>
        </section>

        {/* Trades Table */}
        <section className="bg-surface border border-border rounded-lg">
          <div className="px-4 py-3 border-b border-border">
            <h3 className="text-sm font-semibold text-text-primary">
              Trades {total > 0 && <span className="text-muted font-normal">({total})</span>}
            </h3>
          </div>

          {isLoading ? (
            <div className="p-8 text-center text-muted text-sm animate-pulse">Loading trades...</div>
          ) : trades.length === 0 ? (
            <div className="py-12 text-center text-muted text-sm">No trades found for the selected filters</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Symbol</th>
                    <th>Strategy</th>
                    <th>Direction</th>
                    <th>Qty</th>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>Gross P&L</th>
                    <th>Net P&L</th>
                    <th>Costs</th>
                    <th>Slippage</th>
                    <th>Hold</th>
                    <th>Exit Reason</th>
                    <th>Outcome</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.map((trade) => (
                    <tr
                      key={trade.id}
                      className="cursor-pointer hover:bg-surface-2 transition-colors"
                      onClick={() => setSelectedTrade(trade)}
                    >
                      <td className="font-mono text-xs">{new Date(trade.entryDate).toLocaleDateString('en-IN')}</td>
                      <td className="font-semibold">{trade.symbol}</td>
                      <td className="text-xs capitalize text-text-secondary">{trade.strategy}</td>
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
                      <td className="font-mono tabular-nums text-muted">{formatINR(trade.costs)}</td>
                      <td className="font-mono tabular-nums text-muted">{formatINR(trade.slippage)}</td>
                      <td className="font-mono tabular-nums text-muted">{trade.holdPeriodDays}d</td>
                      <td className="text-xs text-text-secondary max-w-[120px] truncate">{trade.exitReason}</td>
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

      {selectedTrade && (
        <TradeDetailModal trade={selectedTrade} onClose={() => setSelectedTrade(null)} />
      )}
    </div>
  );
}
