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

// REMOVED: MOCK_CORRELATION, five hardcoded pairs (TCS-INFY at 0.91 and so on)
// captioned "based on 90-day rolling window" — a caption describing a
// calculation that never ran. The pairs were not even drawn from the current
// book, so the panel showed correlations for names the account may not hold.
//
// The allocation engine DOES compute a real correlation matrix from price
// history (app/engine/allocated_cycle.estimate_risk_inputs), but it is not yet
// exposed through the API, so the panel states that rather than inventing one.

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

  const activeBreaches = riskState?.activeBreaches ?? [];
  const unavailable = riskState?.unavailable ?? [];

  return (
    <div className="flex flex-col h-full">
      <Header title="Risk Management" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Trading status — the first thing to know is whether it CAN trade. */}
        <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricCard
            title="Portfolio Value"
            value={riskState?.portfolioValue ?? null}
            format="currency"
          />
          <MetricCard
            title="Cash"
            value={riskState?.cash ?? null}
            format="currency"
            subtitle="Uninvested"
          />
          <MetricCard
            title="Largest Position"
            value={
              riskState?.largestPositionPct != null
                ? riskState.largestPositionPct * 100
                : null
            }
            format="percent"
            subtitle={riskState?.largestPositionSymbol ?? 'no positions'}
          />
          <MetricCard
            title="Open Positions"
            value={riskState?.openPositions ?? null}
            format="number"
          />
        </section>

        {/* Blocking conditions. Listed in full rather than summarised: an
            operator needs everything that is stopping trading, not the first
            thing the code happened to check. */}
        {activeBreaches.length > 0 && (
          <section className="bg-loss/10 border border-loss/30 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-loss flex items-center gap-2 mb-3">
              <AlertTriangle className="w-4 h-4" />
              Trading is blocked ({activeBreaches.length})
            </h3>
            <ul className="space-y-1.5">
              {activeBreaches.map((b, i) => (
                <li key={i} className="text-sm text-text-secondary">
                  {b}
                </li>
              ))}
            </ul>
          </section>
        )}

        {riskState && (
          <section className="bg-surface border border-border rounded-lg p-4 text-xs flex flex-wrap gap-x-6 gap-y-2">
            <span className="text-muted">
              Mode <span className="text-text-primary">{riskState.tradingMode}</span>
            </span>
            <span className="text-muted">
              Trading{' '}
              <span className={riskState.tradingPermitted ? 'text-profit' : 'text-loss'}>
                {riskState.tradingPermitted ? 'permitted' : 'blocked'}
              </span>
            </span>
            <span className="text-muted">
              Kill switch{' '}
              <span className={riskState.killSwitchActive ? 'text-loss' : 'text-profit'}>
                {riskState.killSwitchActive ? 'ENGAGED' : 'clear'}
              </span>
            </span>
            {riskState.reconciliationState && (
              <span className="text-muted">
                Reconciliation{' '}
                <span className="text-text-primary">{riskState.reconciliationState}</span>
              </span>
            )}
          </section>
        )}

        {/* Limits. A limit with no current reading is shown as unmeasured
            rather than as a comfortable empty bar. */}
        <section className="bg-surface border border-border rounded-lg p-4">
          <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2 mb-4">
            <Shield className="w-4 h-4 text-accent" />
            Risk limits
          </h3>
          {riskState?.limits?.length ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
              {riskState.limits.map((l) => {
                const pct = l.utilisation != null ? Math.min(l.utilisation * 100, 100) : 0;
                return (
                  <div key={l.name} className="bg-surface-2 rounded p-3">
                    <p className="text-muted mb-1">{l.label}</p>
                    <p className="text-sm font-mono font-semibold text-text-primary mb-2">
                      {l.current != null
                        ? `${l.current.toFixed(2)} / ${l.limit}`
                        : `— / ${l.limit}`}
                    </p>
                    {l.measurable ? (
                      <div className="progress-bar">
                        <div
                          className={clsx(
                            'progress-bar-fill',
                            l.breached ? 'bg-loss' : pct >= 70 ? 'bg-warning' : 'bg-accent',
                          )}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    ) : (
                      <p className="text-2xs text-muted">{l.detail || 'not measurable'}</p>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="text-sm text-muted">Risk limits could not be read.</p>
          )}
        </section>

        {unavailable.length > 0 && (
          <section className="bg-surface border border-border rounded-lg p-4">
            <h3 className="text-sm font-semibold text-text-primary mb-2">
              Not measurable right now
            </h3>
            <ul className="space-y-1 text-xs text-muted">
              {unavailable.map((u, i) => (
                <li key={i}>{u}</li>
              ))}
            </ul>
          </section>
        )}

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
          <div className="py-6 text-center">
            <p className="text-sm text-muted">Not available</p>
            <p className="text-xs text-muted mt-1 max-w-md mx-auto">
              Pairwise correlations are computed by the allocation engine from
              63 sessions of price history, but are not yet exposed through the
              API. The figures previously shown here were hardcoded.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
