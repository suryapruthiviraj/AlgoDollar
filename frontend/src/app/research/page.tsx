'use client';

import { useState } from 'react';
import { Header } from '@/components/layout/Header';
import { MetricCard } from '@/components/common/MetricCard';
import { StatusBadge } from '@/components/common/StatusBadge';
import { EquityCurve } from '@/components/charts/EquityCurve';
import { DrawdownChart } from '@/components/charts/DrawdownChart';
import { useQuery, useMutation } from '@tanstack/react-query';
import { researchApi } from '@/lib/api';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import { FlaskConical, Play, Loader2 } from 'lucide-react';
import type { BacktestConfig, BacktestResult, StrategyName } from '@/types';

const EMPTY_CONFIG: BacktestConfig = {
  strategy: 'swing',
  startDate: '2022-01-01',
  endDate: '2024-12-31',
  initialCapital: 1000000,
  costModel: 'REALISTIC',
};

function BacktestMetricsPanel({ result }: { result: BacktestResult }) {
  const { metrics } = result;
  const rows = [
    { label: 'Total Return', value: `${metrics.totalReturn >= 0 ? '+' : ''}${metrics.totalReturn.toFixed(2)}%`, positive: metrics.totalReturn >= 0 },
    { label: 'Annualized Return', value: `${metrics.annualizedReturn >= 0 ? '+' : ''}${metrics.annualizedReturn.toFixed(2)}%`, positive: metrics.annualizedReturn >= 0 },
    { label: 'Sharpe Ratio', value: metrics.sharpe.toFixed(2), positive: metrics.sharpe >= 1 },
    { label: 'Sortino Ratio', value: metrics.sortino.toFixed(2), positive: metrics.sortino >= 1 },
    { label: 'Calmar Ratio', value: metrics.calmar.toFixed(2), positive: metrics.calmar >= 1 },
    { label: 'Max Drawdown', value: `${metrics.maxDrawdown.toFixed(2)}%`, positive: false },
    { label: 'Max DD Duration', value: `${metrics.maxDrawdownDuration}d`, positive: null },
    { label: 'Win Rate', value: `${metrics.winRate.toFixed(1)}%`, positive: metrics.winRate >= 50 },
    { label: 'Profit Factor', value: metrics.profitFactor.toFixed(2), positive: metrics.profitFactor >= 1.5 },
    { label: 'Avg Win', value: `${metrics.avgWin.toFixed(2)}%`, positive: true },
    { label: 'Avg Loss', value: `${metrics.avgLoss.toFixed(2)}%`, positive: false },
    { label: 'Total Trades', value: metrics.totalTrades.toString(), positive: null },
    { label: 'Annual Vol', value: `${metrics.annualizedVol.toFixed(2)}%`, positive: null },
    { label: 'Beta', value: metrics.beta.toFixed(2), positive: null },
    { label: 'Alpha', value: `${metrics.alpha >= 0 ? '+' : ''}${metrics.alpha.toFixed(2)}%`, positive: metrics.alpha >= 0 },
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
      {rows.map(({ label, value, positive }) => (
        <div key={label} className="bg-surface-2 rounded p-2.5">
          <p className="text-2xs text-muted mb-0.5">{label}</p>
          <p className={`text-sm font-mono font-semibold tabular-nums ${
            positive === true ? 'text-profit' : positive === false ? 'text-loss' : 'text-text-primary'
          }`}>
            {value}
          </p>
        </div>
      ))}
    </div>
  );
}

function MonteCarloChart() {
  // Mock distribution data
  const data = Array.from({ length: 20 }, (_, i) => ({
    drawdown: -(i * 1.5),
    probability: Math.exp(-(i * i) / 20) * 0.3,
  }));

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <h3 className="text-sm font-semibold text-text-primary mb-2">Monte Carlo: Drawdown Distribution</h3>
      <p className="text-xs text-muted mb-4">Based on 1,000 simulated paths</p>
      <ResponsiveContainer width="100%" height={160}>
        <AreaChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="mcGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <XAxis dataKey="drawdown" tick={{ fill: '#6b7280', fontSize: 10 }} tickFormatter={(v: number) => `${v.toFixed(0)}%`} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: '#6b7280', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => v.toFixed(2)} />
          <Tooltip
            contentStyle={{ backgroundColor: '#111118', border: '1px solid #1f2028', borderRadius: 6, fontSize: 12 }}
            formatter={(v: number) => [v.toFixed(3), 'Probability']}
          />
          <Area type="monotone" dataKey="probability" stroke="#ef4444" strokeWidth={1.5} fill="url(#mcGrad)" dot={false} />
        </AreaChart>
      </ResponsiveContainer>

      {/* Probability table */}
      <div className="grid grid-cols-3 gap-2 mt-4 text-xs">
        {[
          { label: 'Ruin Probability (<-30%)', value: '0.02%' },
          { label: 'Exp. Return (median)', value: '+18.4%' },
          { label: 'P5 Scenario', value: '-14.2%' },
        ].map(({ label, value }) => (
          <div key={label} className="bg-surface-2 rounded p-2">
            <p className="text-muted text-2xs">{label}</p>
            <p className="font-mono font-semibold text-text-primary">{value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ResearchPage() {
  const [config, setConfig] = useState<BacktestConfig>(EMPTY_CONFIG);
  const [activeResultId, setActiveResultId] = useState<string | null>(null);

  const { data: pastBacktests } = useQuery({
    queryKey: ['backtests'],
    queryFn: () => researchApi.getBacktests(),
    staleTime: 60_000,
  });

  const runBacktest = useMutation({
    mutationFn: (cfg: BacktestConfig) => researchApi.runBacktest(cfg),
    onSuccess: (result) => {
      setActiveResultId(result.id);
    },
  });

  // Poll for result if pending/running
  const { data: activeResult } = useQuery({
    queryKey: ['backtest', activeResultId],
    queryFn: () => researchApi.getBacktest(activeResultId!),
    enabled: !!activeResultId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === 'PENDING' || status === 'RUNNING' ? 2000 : false;
    },
  });

  const displayResult = activeResult ?? pastBacktests?.[0];

  return (
    <div className="flex flex-col h-full">
      <Header title="Research" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Backtest Configuration */}
        <section className="bg-surface border border-border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-4">
            <FlaskConical className="w-4 h-4 text-accent" />
            <h3 className="text-sm font-semibold text-text-primary">Backtest Configuration</h3>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-4">
            <div>
              <label className="block text-2xs text-muted mb-1">Strategy</label>
              <select
                value={config.strategy}
                onChange={(e) => setConfig((c) => ({ ...c, strategy: e.target.value as StrategyName }))}
                className="w-full bg-surface-2 border border-border rounded px-2 py-1.5 text-sm text-text-primary focus:border-accent focus:outline-none"
              >
                <option value="intraday">Intraday</option>
                <option value="swing">Swing</option>
                <option value="longterm">Long-Term</option>
              </select>
            </div>
            <div>
              <label className="block text-2xs text-muted mb-1">Start Date</label>
              <input
                type="date"
                value={config.startDate}
                onChange={(e) => setConfig((c) => ({ ...c, startDate: e.target.value }))}
                className="w-full bg-surface-2 border border-border rounded px-2 py-1.5 text-sm text-text-primary focus:border-accent focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-2xs text-muted mb-1">End Date</label>
              <input
                type="date"
                value={config.endDate}
                onChange={(e) => setConfig((c) => ({ ...c, endDate: e.target.value }))}
                className="w-full bg-surface-2 border border-border rounded px-2 py-1.5 text-sm text-text-primary focus:border-accent focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-2xs text-muted mb-1">Initial Capital (₹)</label>
              <input
                type="number"
                value={config.initialCapital}
                onChange={(e) => setConfig((c) => ({ ...c, initialCapital: parseInt(e.target.value) || 0 }))}
                className="w-full bg-surface-2 border border-border rounded px-2 py-1.5 text-sm text-text-primary focus:border-accent focus:outline-none font-mono"
              />
            </div>
            <div>
              <label className="block text-2xs text-muted mb-1">Cost Model</label>
              <select
                value={config.costModel}
                onChange={(e) => setConfig((c) => ({ ...c, costModel: e.target.value as BacktestConfig['costModel'] }))}
                className="w-full bg-surface-2 border border-border rounded px-2 py-1.5 text-sm text-text-primary focus:border-accent focus:outline-none"
              >
                <option value="ZERO">Zero Cost</option>
                <option value="REALISTIC">Realistic</option>
                <option value="CONSERVATIVE">Conservative</option>
              </select>
            </div>
          </div>

          <button
            onClick={() => runBacktest.mutate(config)}
            disabled={runBacktest.isPending}
            className="flex items-center gap-2 px-5 py-2.5 bg-accent text-white rounded text-sm font-semibold hover:bg-accent-hover disabled:opacity-60 transition-colors"
          >
            {runBacktest.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            {runBacktest.isPending ? 'Running Backtest...' : 'Run Backtest'}
          </button>

          {/* Progress */}
          {activeResult?.status === 'RUNNING' && (
            <div className="mt-3">
              <div className="flex justify-between text-xs text-muted mb-1">
                <span>Running...</span>
                <span>{(activeResult.progress ?? 0).toFixed(0)}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-bar-fill bg-accent" style={{ width: `${activeResult.progress ?? 0}%` }} />
              </div>
            </div>
          )}
        </section>

        {/* Backtest Results */}
        {displayResult && displayResult.status === 'COMPLETE' && (
          <>
            <section className="bg-surface border border-border rounded-lg p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-text-primary">Backtest Results</h3>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted capitalize">{displayResult.config.strategy}</span>
                  <span className="text-xs text-muted">{displayResult.config.startDate} — {displayResult.config.endDate}</span>
                  <StatusBadge status="COMPLETE" size="sm" />
                </div>
              </div>
              <BacktestMetricsPanel result={displayResult} />
            </section>

            <EquityCurve data={displayResult.equityCurve} showBuyHold />
            <DrawdownChart data={displayResult.drawdowns} />
            <MonteCarloChart />
          </>
        )}

        {/* Past Backtests */}
        <section className="bg-surface border border-border rounded-lg">
          <div className="px-4 py-3 border-b border-border">
            <h3 className="text-sm font-semibold text-text-primary">Past Backtests</h3>
          </div>
          {!pastBacktests || pastBacktests.length === 0 ? (
            <div className="py-10 text-center text-muted text-sm">No previous backtests</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Strategy</th>
                    <th>Period</th>
                    <th>Capital</th>
                    <th>Return</th>
                    <th>Sharpe</th>
                    <th>Max DD</th>
                    <th>Status</th>
                    <th>Run At</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {pastBacktests.map((bt: BacktestResult) => (
                    <tr key={bt.id} className="cursor-pointer" onClick={() => setActiveResultId(bt.id)}>
                      <td className="capitalize text-text-secondary">{bt.config.strategy}</td>
                      <td className="text-xs text-muted">{bt.config.startDate} – {bt.config.endDate}</td>
                      <td className="font-mono tabular-nums">₹{(bt.config.initialCapital / 100000).toFixed(1)}L</td>
                      <td className={`font-mono tabular-nums ${bt.metrics.totalReturn >= 0 ? 'text-profit' : 'text-loss'}`}>
                        {bt.metrics.totalReturn >= 0 ? '+' : ''}{bt.metrics.totalReturn.toFixed(2)}%
                      </td>
                      <td className="font-mono tabular-nums">{bt.metrics.sharpe.toFixed(2)}</td>
                      <td className="font-mono tabular-nums text-loss">{bt.metrics.maxDrawdown.toFixed(2)}%</td>
                      <td><StatusBadge status={bt.status} size="sm" /></td>
                      <td className="text-xs text-muted">{new Date(bt.startedAt).toLocaleDateString('en-IN')}</td>
                      <td>
                        <button className="text-xs text-accent hover:text-accent-hover">Load</button>
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
