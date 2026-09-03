'use client';

import { useState, useMemo } from 'react';
import type { Metadata } from 'next';
import { Header } from '@/components/layout/Header';
import { PortfolioOverview } from '@/components/dashboard/PortfolioOverview';
import { usePortfolio } from '@/hooks/usePortfolio';
import { StatusBadge } from '@/components/common/StatusBadge';
import { clsx } from 'clsx';
import { ArrowUp, ArrowDown, ChevronsUpDown } from 'lucide-react';
import type { Position, StrategyName } from '@/types';

type SortKey = keyof Position;
type SortDir = 'asc' | 'desc';

const STRATEGY_FILTERS = [
  { value: 'all', label: 'All' },
  { value: 'longterm', label: 'Long-Term' },
  { value: 'swing', label: 'Swing' },
  { value: 'intraday', label: 'Intraday' },
] as const;

function formatINR(v: number) {
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(v);
}

function SortIcon({ active, dir }: { active: boolean; dir: SortDir }) {
  if (!active) return <ChevronsUpDown className="w-3 h-3 opacity-30" />;
  return dir === 'asc' ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />;
}

function PositionsTable({ positions }: { positions: Position[] }) {
  const [sortKey, setSortKey] = useState<SortKey>('symbol');
  const [sortDir, setSortDir] = useState<SortDir>('asc');

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  };

  const sorted = useMemo(() => {
    return [...positions].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      const cmp =
        typeof av === 'number' && typeof bv === 'number'
          ? av - bv
          : String(av).localeCompare(String(bv));
      return sortDir === 'asc' ? cmp : -cmp;
    });
  }, [positions, sortKey, sortDir]);

  const th = (key: SortKey, label: string) => (
    <th
      className="px-3 py-2 text-left text-2xs text-muted uppercase tracking-wider cursor-pointer hover:text-text-secondary select-none whitespace-nowrap"
      onClick={() => handleSort(key)}
    >
      <div className="flex items-center gap-1">
        {label}
        <SortIcon active={sortKey === key} dir={sortDir} />
      </div>
    </th>
  );

  if (positions.length === 0) {
    return (
      <div className="text-center py-12 text-muted text-sm">
        No positions found for the selected filter.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="data-table">
        <thead>
          <tr>
            {th('symbol', 'Symbol')}
            {th('qty', 'Qty')}
            {th('avgPrice', 'Avg Price')}
            {th('cmp', 'CMP')}
            {th('marketValue', 'Mkt Value')}
            {th('unrealizedPnl', 'Unreal. P&L')}
            {th('unrealizedPnlPct', 'P&L %')}
            {th('strategy', 'Strategy')}
            {th('sector', 'Sector')}
            {th('portfolioWeight', 'Weight')}
            {th('stopLoss', 'Stop')}
            {th('target', 'Target')}
            <th className="px-3 py-2 text-left text-2xs text-muted uppercase tracking-wider">Action</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((pos) => (
            <tr key={pos.id} className="hover:bg-surface-2 transition-colors">
              <td className="px-3 py-2.5">
                <div className="font-semibold text-text-primary">{pos.symbol}</div>
                <div className="text-2xs text-muted">{pos.exchange}</div>
              </td>
              <td className="px-3 py-2.5 font-mono text-sm tabular-nums">{pos.qty.toLocaleString('en-IN')}</td>
              <td className="px-3 py-2.5 font-mono text-sm tabular-nums">{formatINR(pos.avgPrice)}</td>
              <td className="px-3 py-2.5 font-mono text-sm tabular-nums">{formatINR(pos.cmp)}</td>
              <td className="px-3 py-2.5 font-mono text-sm tabular-nums">{formatINR(pos.marketValue)}</td>
              <td className={clsx('px-3 py-2.5 font-mono text-sm tabular-nums', pos.unrealizedPnl >= 0 ? 'text-profit' : 'text-loss')}>
                {pos.unrealizedPnl >= 0 ? '+' : ''}{formatINR(pos.unrealizedPnl)}
              </td>
              <td className={clsx('px-3 py-2.5 font-mono text-sm tabular-nums', pos.unrealizedPnlPct >= 0 ? 'text-profit' : 'text-loss')}>
                {pos.unrealizedPnlPct >= 0 ? '+' : ''}{pos.unrealizedPnlPct.toFixed(2)}%
              </td>
              <td className="px-3 py-2.5">
                <span className="capitalize text-xs text-text-secondary">{pos.strategy}</span>
              </td>
              <td className="px-3 py-2.5 text-xs text-text-secondary">{pos.sector}</td>
              <td className="px-3 py-2.5 font-mono text-sm tabular-nums text-text-secondary">{pos.portfolioWeight.toFixed(1)}%</td>
              <td className="px-3 py-2.5 font-mono text-sm tabular-nums text-loss">{formatINR(pos.stopLoss)}</td>
              <td className="px-3 py-2.5 font-mono text-sm tabular-nums text-profit">{formatINR(pos.target)}</td>
              <td className="px-3 py-2.5">
                <button className="text-xs text-accent hover:text-accent-hover transition-colors">Details</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function PortfolioPage() {
  const [strategyFilter, setStrategyFilter] = useState<'all' | StrategyName>('all');
  const { positions, isLoading } = usePortfolio(
    strategyFilter !== 'all' ? strategyFilter : undefined,
  );

  return (
    <div className="flex flex-col h-full">
      <Header title="Portfolio" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        <PortfolioOverview />

        <div className="bg-surface border border-border rounded-lg">
          {/* Table header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-border">
            <h3 className="text-sm font-semibold text-text-primary">
              Positions
              {positions.length > 0 && (
                <span className="ml-2 text-xs text-muted font-normal">({positions.length})</span>
              )}
            </h3>

            {/* Strategy filter */}
            <div className="flex gap-1">
              {STRATEGY_FILTERS.map(({ value, label }) => (
                <button
                  key={value}
                  onClick={() => setStrategyFilter(value as 'all' | StrategyName)}
                  className={clsx(
                    'px-3 py-1 rounded text-xs font-medium transition-colors',
                    strategyFilter === value
                      ? 'bg-accent text-white'
                      : 'text-muted hover:text-text-primary hover:bg-surface-2',
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {isLoading ? (
            <div className="p-8 text-center text-muted text-sm animate-pulse">
              Loading positions...
            </div>
          ) : (
            <PositionsTable positions={positions} />
          )}
        </div>
      </div>
    </div>
  );
}
