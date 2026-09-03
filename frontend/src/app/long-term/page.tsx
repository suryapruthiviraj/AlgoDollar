'use client';

import { useState } from 'react';
import { Header } from '@/components/layout/Header';
import { MetricCard } from '@/components/common/MetricCard';
import { StatusBadge } from '@/components/common/StatusBadge';
import { useQuery } from '@tanstack/react-query';
import { portfolioApi, strategiesApi } from '@/lib/api';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { clsx } from 'clsx';
import type { Position } from '@/types';

function formatINR(v: number) {
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(v);
}

// Mock scoring data for long-term positions
function generateScores(symbol: string) {
  const seed = symbol.charCodeAt(0);
  return {
    quality: 40 + (seed % 50),
    growth: 30 + ((seed * 3) % 60),
    valuation: 35 + ((seed * 7) % 55),
    momentum: 25 + ((seed * 11) % 65),
  };
}

function compositeScore(scores: ReturnType<typeof generateScores>): number {
  return (scores.quality * 0.35 + scores.growth * 0.25 + scores.valuation * 0.25 + scores.momentum * 0.15) / 100;
}

function recommendation(score: number): string {
  if (score >= 0.72) return 'BUY';
  if (score >= 0.55) return 'HOLD';
  if (score >= 0.4) return 'REDUCE';
  return 'SELL';
}

function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-20 text-muted shrink-0">{label}</span>
      <div className="flex-1 h-1.5 bg-border rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${value}%`, backgroundColor: color }} />
      </div>
      <span className="w-6 text-right font-mono text-text-secondary">{value}</span>
    </div>
  );
}

function PositionDetail({ pos }: { pos: Position }) {
  const scores = generateScores(pos.symbol);
  const composite = compositeScore(scores);
  const rec = recommendation(composite);

  const recColor = rec === 'BUY' ? '#22c55e' : rec === 'HOLD' ? '#3b82f6' : rec === 'REDUCE' ? '#f59e0b' : '#ef4444';

  const chartData = [
    { name: 'Quality', value: scores.quality, fill: '#3b82f6' },
    { name: 'Growth', value: scores.growth, fill: '#22c55e' },
    { name: 'Valuation', value: scores.valuation, fill: '#8b5cf6' },
    { name: 'Momentum', value: scores.momentum, fill: '#f59e0b' },
  ];

  return (
    <div className="bg-surface-2 rounded-lg p-4 space-y-3">
      <div className="flex items-start justify-between">
        <div>
          <div className="font-semibold text-text-primary">{pos.symbol}</div>
          <div className="text-xs text-muted">{pos.sector}</div>
        </div>
        <div className="text-right">
          <div
            className="text-xs font-bold px-2 py-1 rounded"
            style={{ color: recColor, backgroundColor: `${recColor}20`, border: `1px solid ${recColor}40` }}
          >
            {rec}
          </div>
          <div className="text-xs text-muted mt-1">Score: {(composite * 100).toFixed(0)}/100</div>
        </div>
      </div>

      {/* Score bars */}
      <div className="space-y-1.5">
        <ScoreBar label="Quality" value={scores.quality} color="#3b82f6" />
        <ScoreBar label="Growth" value={scores.growth} color="#22c55e" />
        <ScoreBar label="Valuation" value={scores.valuation} color="#8b5cf6" />
        <ScoreBar label="Momentum" value={scores.momentum} color="#f59e0b" />
      </div>

      {/* Recommendation explanation */}
      <p className="text-2xs text-text-secondary">
        {rec === 'BUY' && 'Strong fundamentals with attractive valuation and positive momentum. Consider adding to position.'}
        {rec === 'HOLD' && 'Solid quality and growth profile. Current position size appropriate. Monitor for rebalancing triggers.'}
        {rec === 'REDUCE' && 'Elevated valuation relative to growth expectations. Consider trimming position on strength.'}
        {rec === 'SELL' && 'Deteriorating fundamentals and momentum. Exit position systematically.'}
      </p>
    </div>
  );
}

export default function LongTermPage() {
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);

  const { data: positions, isLoading } = useQuery({
    queryKey: ['portfolio', 'positions', 'longterm'],
    queryFn: () => portfolioApi.getPositions('longterm'),
    refetchInterval: 60_000,
  });

  const { data: performance } = useQuery({
    queryKey: ['strategies', 'longterm', 'performance'],
    queryFn: () => strategiesApi.getPerformance('longterm'),
    staleTime: 120_000,
  });

  const ltPositions = positions ?? [];

  // Build comparison chart data
  const chartData = ltPositions.map((pos) => {
    const scores = generateScores(pos.symbol);
    return { symbol: pos.symbol, score: Math.round(compositeScore(scores) * 100) };
  }).sort((a, b) => b.score - a.score);

  return (
    <div className="flex flex-col h-full">
      <Header title="Long-Term Holdings" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Performance Stats */}
        <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricCard title="Total Return" value={performance?.totalReturn ?? null} format="percent" loading={!performance} />
          <MetricCard title="1Y Return" value={performance?.return1y ?? null} format="percent" loading={!performance} />
          <MetricCard title="Sharpe" value={performance?.sharpe ?? null} format="number" loading={!performance} />
          <MetricCard title="Max Drawdown" value={performance?.maxDrawdown ?? null} format="percent" invertColors loading={!performance} />
        </section>

        {/* Holdings Table */}
        <section className="bg-surface border border-border rounded-lg">
          <div className="px-4 py-3 border-b border-border">
            <h3 className="text-sm font-semibold text-text-primary">
              Long-Term Holdings ({ltPositions.length})
            </h3>
          </div>

          {isLoading ? (
            <div className="p-8 text-center text-muted text-sm animate-pulse">Loading holdings...</div>
          ) : ltPositions.length === 0 ? (
            <div className="py-10 text-center text-muted text-sm">No long-term holdings</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Quality</th>
                    <th>Growth</th>
                    <th>Valuation</th>
                    <th>Momentum</th>
                    <th>Composite</th>
                    <th>Rec.</th>
                    <th>Invested</th>
                    <th>Current Value</th>
                    <th>P&L</th>
                    <th>Details</th>
                  </tr>
                </thead>
                <tbody>
                  {ltPositions.map((pos: Position) => {
                    const scores = generateScores(pos.symbol);
                    const composite = compositeScore(scores);
                    const rec = recommendation(composite);
                    const recColor = rec === 'BUY' ? 'text-profit' : rec === 'HOLD' ? 'text-accent' : rec === 'REDUCE' ? 'text-warning' : 'text-loss';

                    return (
                      <tr key={pos.id} className={clsx(selectedSymbol === pos.symbol && 'bg-surface-2')}>
                        <td>
                          <div className="font-semibold text-text-primary">{pos.symbol}</div>
                          <div className="text-2xs text-muted">{pos.sector}</div>
                        </td>
                        <td className="font-mono tabular-nums">{scores.quality}</td>
                        <td className="font-mono tabular-nums">{scores.growth}</td>
                        <td className="font-mono tabular-nums">{scores.valuation}</td>
                        <td className="font-mono tabular-nums">{scores.momentum}</td>
                        <td className="font-mono tabular-nums">
                          <div className="flex items-center gap-1.5">
                            <div className="w-10 h-1.5 bg-border rounded-full overflow-hidden">
                              <div className="h-full bg-accent" style={{ width: `${composite * 100}%` }} />
                            </div>
                            {(composite * 100).toFixed(0)}
                          </div>
                        </td>
                        <td className={clsx('font-semibold text-xs', recColor)}>{rec}</td>
                        <td className="font-mono tabular-nums">{formatINR(pos.avgPrice * pos.qty)}</td>
                        <td className="font-mono tabular-nums">{formatINR(pos.marketValue)}</td>
                        <td className={clsx('font-mono tabular-nums', pos.unrealizedPnl >= 0 ? 'text-profit' : 'text-loss')}>
                          {pos.unrealizedPnl >= 0 ? '+' : ''}{pos.unrealizedPnlPct.toFixed(2)}%
                        </td>
                        <td>
                          <button
                            className="text-xs text-accent hover:text-accent-hover"
                            onClick={() => setSelectedSymbol(selectedSymbol === pos.symbol ? null : pos.symbol)}
                          >
                            {selectedSymbol === pos.symbol ? 'Hide' : 'View'}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* Selected Position Detail */}
        {selectedSymbol && (
          <section>
            <h3 className="text-sm font-semibold text-text-primary mb-3">Scorecard: {selectedSymbol}</h3>
            {ltPositions
              .filter((p) => p.symbol === selectedSymbol)
              .map((pos) => (
                <PositionDetail key={pos.id} pos={pos} />
              ))}
          </section>
        )}

        {/* Composite Score Comparison */}
        {chartData.length > 0 && (
          <section className="bg-surface border border-border rounded-lg p-4">
            <h3 className="text-sm font-semibold text-text-primary mb-4">Composite Score Comparison</h3>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <XAxis dataKey="symbol" tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 100]} tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#111118', border: '1px solid #1f2028', borderRadius: 6, fontSize: 12 }}
                  labelStyle={{ color: '#f1f5f9' }}
                  cursor={{ fill: 'rgba(59,130,246,0.08)' }}
                />
                <Bar dataKey="score" radius={[3, 3, 0, 0]}>
                  {chartData.map((entry) => (
                    <Cell
                      key={entry.symbol}
                      fill={entry.score >= 72 ? '#22c55e' : entry.score >= 55 ? '#3b82f6' : entry.score >= 40 ? '#f59e0b' : '#ef4444'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </section>
        )}
      </div>
    </div>
  );
}
