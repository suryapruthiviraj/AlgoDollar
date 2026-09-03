'use client';

import { useState } from 'react';
import { Header } from '@/components/layout/Header';
import { EquityCurve } from '@/components/charts/EquityCurve';
import { DrawdownChart } from '@/components/charts/DrawdownChart';
import { MetricCard } from '@/components/common/MetricCard';
import { useQuery } from '@tanstack/react-query';
import { portfolioApi } from '@/lib/api';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ScatterChart,
} from 'recharts';
import { clsx } from 'clsx';

type Period = '1D' | '1W' | '1M' | '3M' | '6M' | '1Y' | '3Y' | 'ALL';
const PERIODS: Period[] = ['1D', '1W', '1M', '3M', '6M', '1Y', '3Y', 'ALL'];

// Generate monthly returns heatmap data
function generateMonthlyReturns() {
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const years = [2022, 2023, 2024, 2025];
  return years.map((year) => ({
    year,
    months: months.map((month, mi) => ({
      month,
      return: (Math.random() - 0.45) * 8,
      yearMonth: `${year}-${String(mi + 1).padStart(2, '0')}`,
    })),
  }));
}

// Generate strategy comparison data
const strategyData = [
  { strategy: 'Long-Term', return: 18.4, sharpe: 1.42, maxDD: -8.2 },
  { strategy: 'Swing', return: 24.1, sharpe: 1.18, maxDD: -12.5 },
  { strategy: 'Intraday', return: 31.7, sharpe: 0.89, maxDD: -18.3 },
];

// Cost analysis data
const costData = Array.from({ length: 12 }, (_, i) => ({
  month: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][i],
  costs: 800 + Math.random() * 400,
  grossPnl: 3000 + (Math.random() - 0.4) * 2000,
}));

function MonthlyHeatmap() {
  const data = generateMonthlyReturns();
  const allReturns = data.flatMap((y) => y.months.map((m) => Math.abs(m.return)));
  const maxAbs = Math.max(...allReturns, 1);

  const cellBg = (ret: number) => {
    const intensity = Math.min(Math.abs(ret) / maxAbs, 1);
    if (ret > 0) return `rgba(34, 197, 94, ${0.15 + intensity * 0.6})`;
    if (ret < 0) return `rgba(239, 68, 68, ${0.15 + intensity * 0.6})`;
    return 'rgba(107, 114, 128, 0.1)';
  };

  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr>
            <th className="text-left text-muted py-1 pr-3 w-12">Year</th>
            {months.map((m) => (
              <th key={m} className="text-muted py-1 px-1 text-center">{m}</th>
            ))}
            <th className="text-muted py-1 pl-2 text-right">YTD</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row) => {
            const ytd = row.months.reduce((acc, m) => acc * (1 + m.return / 100), 1) - 1;
            return (
              <tr key={row.year}>
                <td className="py-1 pr-3 font-semibold text-text-secondary">{row.year}</td>
                {row.months.map((m) => (
                  <td
                    key={m.month}
                    className="py-1 px-0.5 text-center rounded"
                    title={`${m.month} ${row.year}: ${m.return.toFixed(2)}%`}
                  >
                    <div
                      className="rounded py-1 px-1"
                      style={{ backgroundColor: cellBg(m.return) }}
                    >
                      <span className={clsx('font-mono', m.return >= 0 ? 'text-profit' : 'text-loss')}>
                        {m.return.toFixed(1)}%
                      </span>
                    </div>
                  </td>
                ))}
                <td className={clsx('py-1 pl-2 text-right font-mono font-semibold', ytd >= 0 ? 'text-profit' : 'text-loss')}>
                  {(ytd * 100) >= 0 ? '+' : ''}{(ytd * 100).toFixed(1)}%
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function AnalyticsPage() {
  const [period, setPeriod] = useState<Period>('1Y');

  const { data: overview } = useQuery({
    queryKey: ['portfolio', 'performance', period],
    queryFn: () => portfolioApi.getPerformance(period),
    staleTime: 60_000,
  });

  const riskMetrics = [
    { label: 'Sharpe Ratio', value: overview?.sharpeRatio ?? 1.24 },
    { label: 'Sortino Ratio', value: overview?.sortinoRatio ?? 1.68 },
    { label: 'Max Drawdown', value: overview?.maxDrawdown ?? -9.4 },
    { label: 'Calmar Ratio', value: (overview?.totalReturnPct ?? 22.1) / Math.abs(overview?.maxDrawdown ?? 9.4) },
    { label: 'Annual Vol', value: overview?.portfolioVol ?? 14.2 },
    { label: 'Beta (NIFTY)', value: overview?.beta ?? 0.78 },
  ];

  return (
    <div className="flex flex-col h-full">
      <Header title="Analytics" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Period Selector */}
        <div className="flex gap-1">
          {PERIODS.map((p) => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              className={clsx(
                'px-3 py-1.5 rounded text-xs font-medium transition-colors',
                period === p
                  ? 'bg-accent text-white'
                  : 'text-muted hover:text-text-primary hover:bg-surface',
              )}
            >
              {p}
            </button>
          ))}
        </div>

        {/* Equity Curve */}
        <EquityCurve showBuyHold />

        {/* Drawdown */}
        <DrawdownChart />

        {/* Monthly Returns Heatmap */}
        <section className="bg-surface border border-border rounded-lg p-4">
          <h3 className="text-sm font-semibold text-text-primary mb-4">Monthly Returns Heatmap</h3>
          <MonthlyHeatmap />
        </section>

        {/* Strategy Comparison */}
        <section className="bg-surface border border-border rounded-lg p-4">
          <h3 className="text-sm font-semibold text-text-primary mb-4">Strategy Returns Comparison</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={strategyData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <XAxis dataKey="strategy" tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `${v}%`} />
              <Tooltip
                contentStyle={{ backgroundColor: '#111118', border: '1px solid #1f2028', borderRadius: 6, fontSize: 12 }}
                labelStyle={{ color: '#f1f5f9' }}
                formatter={(v: number) => [`${v.toFixed(1)}%`, 'Return']}
              />
              <Bar dataKey="return" radius={[3, 3, 0, 0]}>
                {strategyData.map((entry) => (
                  <Cell key={entry.strategy} fill={entry.return >= 0 ? '#22c55e' : '#ef4444'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </section>

        {/* Risk Metrics Table */}
        <section className="bg-surface border border-border rounded-lg p-4">
          <h3 className="text-sm font-semibold text-text-primary mb-4">Risk Metrics</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {riskMetrics.map(({ label, value }) => (
              <div key={label} className="bg-surface-2 rounded-lg p-3">
                <p className="text-xs text-muted mb-1">{label}</p>
                <p className={clsx(
                  'text-lg font-mono font-semibold tabular-nums',
                  value < 0 ? 'text-loss' : value > 0 ? 'text-text-primary' : 'text-muted',
                )}>
                  {value < 0 ? '' : ''}{value.toFixed(2)}{label.includes('Ratio') ? '' : '%'}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Cost Analysis */}
        <section className="bg-surface border border-border rounded-lg p-4">
          <h3 className="text-sm font-semibold text-text-primary mb-4">Cost Analysis</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-muted mb-3">Monthly Costs vs Gross P&L</p>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={costData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                  <XAxis dataKey="month" tick={{ fill: '#6b7280', fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => `₹${(v / 1000).toFixed(0)}k`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#111118', border: '1px solid #1f2028', borderRadius: 6, fontSize: 12 }}
                    formatter={(v: number, name: string) => [`₹${v.toFixed(0)}`, name === 'costs' ? 'Costs' : 'Gross P&L']}
                  />
                  <Bar dataKey="grossPnl" name="grossPnl" fill="#3b82f620" stroke="#3b82f6" strokeWidth={1} radius={[2, 2, 0, 0]} />
                  <Bar dataKey="costs" name="costs" fill="#ef444440" stroke="#ef4444" strokeWidth={1} radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div>
              <p className="text-xs text-muted mb-3">Cost as % of Gross P&L</p>
              <div className="space-y-2">
                {costData.slice(0, 6).map((d) => {
                  const pct = d.grossPnl > 0 ? (d.costs / d.grossPnl) * 100 : 0;
                  return (
                    <div key={d.month} className="flex items-center gap-3 text-xs">
                      <span className="text-muted w-8">{d.month}</span>
                      <div className="flex-1 h-1.5 bg-border rounded-full overflow-hidden">
                        <div
                          className={clsx('h-full', pct > 30 ? 'bg-loss' : pct > 15 ? 'bg-warning' : 'bg-accent')}
                          style={{ width: `${Math.min(pct, 100)}%` }}
                        />
                      </div>
                      <span className="font-mono w-10 text-right text-text-secondary">{pct.toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
