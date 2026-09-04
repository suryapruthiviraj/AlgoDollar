'use client';

import { Header } from '@/components/layout/Header';
import { MetricCard } from '@/components/common/MetricCard';
import { useQuery } from '@tanstack/react-query';
import { riskApi, portfolioApi } from '@/lib/api';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
} from 'recharts';
import { clsx } from 'clsx';
import { Shield, AlertTriangle } from 'lucide-react';
import type { RiskContribution, Position } from '@/types';

function formatINR(v: number) {
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(v);
}

// Mock sector concentration from positions
function computeSectorConcentration(positions: Position[]) {
  const sectorMap: Record<string, number> = {};
  const total = positions.reduce((s, p) => s + p.marketValue, 0);
  for (const pos of positions) {
    sectorMap[pos.sector] = (sectorMap[pos.sector] ?? 0) + pos.marketValue;
  }
  return Object.entries(sectorMap).map(([sector, value]) => ({
    sector,
    value,
    pct: total > 0 ? (value / total) * 100 : 0,
  }));
}

// Mock correlation matrix (in reality would come from API)
const MOCK_CORRELATION = [
  { pair: 'RELIANCE-ONGC', correlation: 0.82 },
  { pair: 'HDFC-ICICI', correlation: 0.78 },
  { pair: 'TCS-INFY', correlation: 0.91 },
  { pair: 'BHARTI-VODAIDEA', correlation: 0.65 },
  { pair: 'MARUTI-M&M', correlation: 0.73 },
];

const SECTOR_COLORS = ['#3b82f6', '#22c55e', '#8b5cf6', '#f59e0b', '#06b6d4', '#ec4899', '#84cc16', '#f97316'];

function CorrelationRow({ pair, value }: { pair: string; value: number }) {
  const intensity = Math.abs(value);
  const color = value > 0.7 ? 'bg-loss/20 text-loss' : value > 0.4 ? 'bg-warning/20 text-warning' : 'bg-profit/20 text-profit';
  return (
    <div className="flex items-center justify-between text-xs py-1.5 border-b border-border last:border-0">
      <span className="text-text-secondary font-mono">{pair}</span>
      <div className="flex items-center gap-2">
        <div className="w-20 h-1.5 bg-border rounded-full overflow-hidden">
          <div className="h-full rounded-full bg-loss" style={{ width: `${intensity * 100}%` }} />
        </div>
        <span className={clsx('px-1.5 py-0.5 rounded text-2xs font-semibold', color)}>
          {value.toFixed(2)}
        </span>
      </div>
    </div>
  );
}

export default function RiskPage() {
  const { data: riskState } = useQuery({
    queryKey: ['risk', 'state'],
    queryFn: () => riskApi.getState(),
    refetchInterval: 15_000,
  });

  const { data: riskLimits } = useQuery({
    queryKey: ['risk', 'limits'],
    queryFn: () => riskApi.getLimits(),
    staleTime: 120_000,
  });

  const { data: positions } = useQuery({
    queryKey: ['portfolio', 'positions'],
    queryFn: () => portfolioApi.getPositions(),
    refetchInterval: 30_000,
  });

  const sectorConcentration = computeSectorConcentration(positions ?? []);

  // REMOVED: a "Risk Contribution by Position" chart whose bars were
  // fabricated. It read:
  //
  //     riskContributionPct: pos.portfolioWeight * 0.8 + Math.random() * 2,
  //     beta:                0.7 + Math.random() * 0.8,
  //
  // so a risk dashboard was drawing invented risk contributions and betas next
  // to real position data, re-rolled on every render. Marginal risk
  // contribution needs a covariance matrix and beta needs a regression against
  // an index; the backend computes neither, so there is nothing real to show.
  //
  // This is the same call already made for the /research backtest and /markets
  // sector endpoints, which now return 501 rather than invent numbers: an empty
  // panel that says why is honest, a plausible-looking chart is not. Restore
  // the panel when the backend can supply measured values.

  const activeBreaches = riskState?.activeBreach ?? [];

  return (
    <div className="flex flex-col h-full">
      <Header title="Risk Management" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Risk State Overview */}
        <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricCard
            title="Portfolio Drawdown"
            value={riskState?.portfolioDrawdown ?? null}
            format="percent"
            invertColors
          />
          <MetricCard
            title="VaR 95%"
            value={riskState?.var95 ?? null}
            format="currency"
            invertColors
            subtitle="1-day"
          />
          <MetricCard
            title="CVaR 95%"
            value={riskState?.cvar95 ?? null}
            format="currency"
            invertColors
            subtitle="Expected shortfall"
          />
          <MetricCard
            title="Concentration"
            value={riskState?.concentrationRisk ?? null}
            format="percent"
            subtitle="Top position weight"
          />
        </section>

        {/* Active Breaches */}
        {activeBreaches.length > 0 && (
          <section className="bg-loss/10 border border-loss/30 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-loss flex items-center gap-2 mb-3">
              <AlertTriangle className="w-4 h-4" />
              Active Risk Limit Breaches ({activeBreaches.length})
            </h3>
            <div className="space-y-2">
              {activeBreaches.map((breach, i) => (
                <div key={i} className="flex items-start gap-3 text-sm">
                  <span className={clsx(
                    'text-xs px-1.5 py-0.5 rounded font-semibold shrink-0',
                    breach.severity === 'CRITICAL' ? 'bg-loss text-white' : 'bg-warning/20 text-warning',
                  )}>
                    {breach.severity}
                  </span>
                  <span className="text-text-secondary">{breach.message}</span>
                  <span className="text-muted text-xs shrink-0 ml-auto">
                    {new Date(breach.triggeredAt).toLocaleTimeString('en-IN')}
                  </span>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Risk Limits Reference */}
        <section className="bg-surface border border-border rounded-lg p-4">
          <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2 mb-4">
            <Shield className="w-4 h-4 text-accent" />
            Risk Limits Configuration
          </h3>
          {riskLimits ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
              {[
                { label: 'Max Portfolio Drawdown', value: `${riskLimits.maxPortfolioDrawdown}%`, used: Math.abs(riskState?.portfolioDrawdown ?? 0), max: riskLimits.maxPortfolioDrawdown },
                { label: 'Max Daily Loss', value: formatINR(riskLimits.maxDailyLoss), used: Math.abs(riskState?.dailyLoss ?? 0), max: riskLimits.maxDailyLoss },
                { label: 'Max Weekly Loss', value: formatINR(riskLimits.maxWeeklyLoss), used: Math.abs(riskState?.weeklyLoss ?? 0), max: riskLimits.maxWeeklyLoss },
                { label: 'Max Position Size', value: `${riskLimits.maxPositionSize}%`, used: 0, max: 1 },
                { label: 'Max Sector Conc.', value: `${riskLimits.maxSectorConcentration}%`, used: 0, max: 1 },
                { label: 'Max Correlation', value: riskLimits.maxCorrelation.toFixed(2), used: 0, max: 1 },
              ].map(({ label, value, used, max }) => {
                const pct = max > 0 ? Math.min((used / max) * 100, 100) : 0;
                return (
                  <div key={label} className="bg-surface-2 rounded p-3">
                    <p className="text-muted mb-1">{label}</p>
                    <p className="text-sm font-mono font-semibold text-text-primary mb-2">{value}</p>
                    <div className="progress-bar">
                      <div
                        className={clsx('progress-bar-fill', pct >= 90 ? 'bg-loss' : pct >= 70 ? 'bg-warning' : 'bg-accent')}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-sm text-muted">Loading risk limits...</div>
          )}
        </section>

        {/* Risk Contribution + Sector Concentration */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Risk Contribution by Position — see the note at the top of this
              file. The chart that stood here plotted Math.random(), so it is
              deliberately blank rather than plausible. */}
          <div className="bg-surface border border-border rounded-lg p-4">
            <h3 className="text-sm font-semibold text-text-primary mb-4">Risk Contribution by Position</h3>
            <div className="flex flex-col items-center justify-center text-center py-8 gap-1">
              <span className="text-sm text-muted">Not available</span>
              <span className="text-xs text-muted max-w-xs">
                Marginal risk contribution requires a position covariance matrix,
                which this system does not yet compute.
              </span>
            </div>
          </div>

          {/* Sector Concentration */}
          <div className="bg-surface border border-border rounded-lg p-4">
            <h3 className="text-sm font-semibold text-text-primary mb-4">Sector Concentration</h3>
            {sectorConcentration.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={sectorConcentration}
                    dataKey="pct"
                    nameKey="sector"
                    cx="50%"
                    cy="50%"
                    innerRadius="45%"
                    outerRadius="75%"
                    paddingAngle={2}
                  >
                    {sectorConcentration.map((_, i) => (
                      <Cell key={i} fill={SECTOR_COLORS[i % SECTOR_COLORS.length]} stroke="transparent" />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ backgroundColor: '#111118', border: '1px solid #1f2028', borderRadius: 6, fontSize: 12 }}
                    formatter={(v: number) => [`${v.toFixed(1)}%`, 'Weight']}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-center text-muted text-sm py-8">No position data</div>
            )}
          </div>
        </section>

        {/* Correlation Matrix */}
        <section className="bg-surface border border-border rounded-lg p-4">
          <h3 className="text-sm font-semibold text-text-primary mb-4">Top Position Correlations</h3>
          <div className="space-y-0">
            {MOCK_CORRELATION.map((c) => (
              <CorrelationRow key={c.pair} pair={c.pair} value={c.correlation} />
            ))}
          </div>
          <p className="text-2xs text-muted mt-3">Correlations based on 90-day rolling window. Values above 0.7 flagged as high.</p>
        </section>
      </div>
    </div>
  );
}
