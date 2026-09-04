'use client';

import { usePortfolio } from '@/hooks/usePortfolio';
import { clsx } from 'clsx';
import { LineChart, Line, ResponsiveContainer } from 'recharts';

function formatINR(value: number): string {
  const abs = Math.abs(value);
  const formatted = new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0,
  }).format(abs);
  return value < 0 ? `-${formatted}` : formatted;
}

interface PnLRowProps {
  label: string;
  /** Undefined means the backend did not report it — rendered as "—", not 0. */
  net: number | undefined;
}

function PnLRow({ label, net }: PnLRowProps) {
  const known = typeof net === 'number' && Number.isFinite(net);
  const isPositive = known && (net as number) >= 0;

  return (
    <div className="flex items-center gap-4 py-3 border-b border-border last:border-b-0">
      <div className="w-20 shrink-0">
        <p className="text-xs text-muted">{label}</p>
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-baseline gap-2">
          <span
            className={clsx(
              'text-sm font-mono font-semibold tabular-nums',
              isPositive ? 'text-profit' : 'text-loss',
            )}
          >
            {!known ? '—' : `${isPositive ? '+' : ''}${formatINR(net as number)}`}
          </span>
          {!known && (
            <span className="text-2xs text-muted">
              not reported
            </span>
          )}
        </div>
        {/* The Gross / Costs sub-line is gone: gross was net inflated by an
            assumed 0.05%, so "Costs" was that assumption displayed as a
            measurement. */}
      </div>

    </div>
  );
}

// REMOVED: mockSparkline(), which drew a random walk shaped to match whichever
// direction the P&L happened to be. It made a made-up trend look like history.

export function PnLCard() {
  const { overview, isLoading } = usePortfolio();

  if (isLoading) {
    return (
      <div className="bg-surface border border-border rounded-lg p-4 animate-pulse">
        <div className="h-4 w-20 bg-border rounded mb-4" />
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-12 w-full bg-border rounded mb-2" />
        ))}
      </div>
    );
  }

  // Only what the backend actually reports. The endpoint returns today_pnl,
  // monthly_pnl, unrealized_pnl, realized_pnl and total_return_pct — there is
  // no weekly figure, so none is shown.
  const today = overview?.todayPnl;
  const monthly = overview?.monthlyPnl;
  const realized = overview?.realizedPnl;
  const unrealized = overview?.unrealizedPnl;

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <h3 className="text-sm font-semibold text-text-primary mb-1">P&L Breakdown</h3>
      {/*
        The GROSS column is gone. It was computed as `net * 1.0005` — net
        inflated by an assumed 0.05% cost, presented next to the real net figure
        as though both had been measured. Costs ARE recorded per fill and are
        available on the trades endpoint; when they are aggregated here the
        column can come back as a measurement.
      */}
      <p className="text-2xs text-muted mb-3">Net, after transaction costs</p>

      <PnLRow label="Today" net={today} />
      <PnLRow label="This Month" net={monthly} />
      <PnLRow label="Realised" net={realized} />
      <PnLRow label="Unrealised" net={unrealized} />
    </div>
  );
}
