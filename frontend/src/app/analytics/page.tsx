'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { clsx } from 'clsx';
import { Header } from '@/components/layout/Header';
import { EquityCurve } from '@/components/charts/EquityCurve';
import { DrawdownChart } from '@/components/charts/DrawdownChart';
import { portfolioApi } from '@/lib/api';

/**
 * Performance analytics, from the backend's own numbers.
 *
 * WHAT WAS REMOVED AND WHY
 * This page previously rendered:
 *   - a monthly-returns heatmap built from Math.random()
 *   - a per-strategy comparison chart with hardcoded returns and Sharpes
 *   - a cost-vs-P&L series built from Math.random()
 *   - risk metrics whose "fallbacks" (?? 1.24, ?? -9.4, ?? 0.78) were invented
 *     numbers displayed whenever the API returned nothing — which was always,
 *     because the client was reading the wrong response shape.
 *
 * All of it looked exactly like measured performance. None of it was. A panel
 * that says "not computed" is useful; one showing a plausible Sharpe of 1.24
 * for a system with no validated strategy is actively misleading.
 */

type Period = '1D' | '1W' | '1M' | '3M' | '6M' | '1Y' | '3Y' | 'ALL';
const PERIODS: Period[] = ['1D', '1W', '1M', '3M', '6M', '1Y', '3Y', 'ALL'];

/** A panel the backend cannot fill yet, saying so instead of inventing it. */
function NotComputed({ title, reason }: { title: string; reason: string }) {
  return (
    <section className="bg-surface border border-border rounded-lg p-4">
      <h3 className="text-sm font-semibold text-text-primary mb-2">{title}</h3>
      <div className="py-6 text-center">
        <p className="text-sm text-muted">Not computed</p>
        <p className="text-xs text-muted mt-1 max-w-md mx-auto">{reason}</p>
      </div>
    </section>
  );
}

function Metric({
  label,
  value,
  format = 'ratio',
}: {
  label: string;
  value: number | null | undefined;
  format?: 'ratio' | 'percent';
}) {
  // null and undefined render as "—", never as 0. A Sharpe of 0.00 shown
  // because nothing was measured reads as a real, poor result.
  const known = typeof value === 'number' && Number.isFinite(value);
  const text = !known
    ? '—'
    : format === 'percent'
      ? `${(value as number).toFixed(2)}%`
      : (value as number).toFixed(2);

  return (
    <div className="bg-surface border border-border rounded-lg p-3">
      <p className="text-xs text-muted">{label}</p>
      <p
        className={clsx(
          'text-lg font-mono mt-1',
          !known
            ? 'text-muted'
            : (value as number) >= 0
              ? 'text-text-primary'
              : 'text-loss',
        )}
      >
        {text}
      </p>
      {!known && <p className="text-[10px] text-muted mt-0.5">not measured</p>}
    </div>
  );
}

export default function AnalyticsPage() {
  const [period, setPeriod] = useState<Period>('1Y');

  // The OVERVIEW carries the risk statistics. `getPerformance` returns the
  // equity curve — this page used to read Sharpe and drawdown off that array,
  // where those fields have never existed.
  const { data: overview, isLoading } = useQuery({
    queryKey: ['portfolio', 'overview'],
    queryFn: () => portfolioApi.getOverview(),
    staleTime: 60_000,
  });

  const calmar =
    overview && overview.drawdownPct
      ? overview.totalReturnPct / Math.abs(overview.drawdownPct)
      : null;

  return (
    <div className="flex flex-col h-full">
      <Header title="Analytics" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
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

        <section>
          <h3 className="text-sm font-semibold text-text-primary mb-3">
            Risk-adjusted performance
          </h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <Metric label="Sharpe" value={isLoading ? null : overview?.sharpe} />
            <Metric label="Sortino" value={isLoading ? null : overview?.sortino} />
            <Metric
              label="Max Drawdown"
              value={isLoading ? null : overview?.drawdownPct}
              format="percent"
            />
            <Metric label="Calmar" value={isLoading ? null : calmar} />
            <Metric
              label="Annual Vol"
              value={isLoading ? null : overview?.volatility}
              format="percent"
            />
            {/* Beta needs a regression against the index, which the backend
                does not compute. Shown as unmeasured rather than guessed. */}
            <Metric label="Beta (NIFTY)" value={null} />
          </div>
        </section>

        <EquityCurve showBuyHold />

        <DrawdownChart />

        <NotComputed
          title="Monthly Returns Heatmap"
          reason="Requires a persisted monthly return series. The backend stores trades and positions but does not yet aggregate returns by month."
        />

        <NotComputed
          title="Strategy Returns Comparison"
          reason="Requires per-strategy P&L attribution over time. Trades carry a strategy label, but returns are not yet aggregated per sleeve."
        />

        <NotComputed
          title="Transaction Costs vs Gross P&L"
          reason="Costs are recorded per fill and available on the trades endpoint, but are not yet aggregated into a monthly series."
        />
      </div>
    </div>
  );
}
