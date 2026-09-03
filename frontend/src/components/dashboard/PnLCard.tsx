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
  gross: number;
  net: number;
  grossPct?: number;
  sparkline?: number[];
}

function PnLRow({ label, gross, net, grossPct, sparkline }: PnLRowProps) {
  const isPositive = net >= 0;
  const sparkData = sparkline?.map((v) => ({ v })) ?? [];

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
            {isPositive ? '+' : ''}{formatINR(net)}
          </span>
          {grossPct !== undefined && (
            <span className={clsx('text-2xs font-mono', isPositive ? 'text-profit' : 'text-loss')}>
              ({grossPct >= 0 ? '+' : ''}{grossPct.toFixed(2)}%)
            </span>
          )}
        </div>
        <div className="flex gap-3 text-2xs text-muted mt-0.5">
          <span>Gross: {formatINR(gross)}</span>
          <span>Net: {formatINR(net)}</span>
          {gross !== 0 && (
            <span>Costs: {formatINR(gross - net)}</span>
          )}
        </div>
      </div>

      {sparkline && sparkline.length > 0 && (
        <div className="w-16 h-8 shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={sparkData}>
              <Line
                type="monotone"
                dataKey="v"
                stroke={isPositive ? '#22c55e' : '#ef4444'}
                strokeWidth={1.5}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

// Mock sparkline generator
function mockSparkline(trend: 'up' | 'down'): number[] {
  const points: number[] = [];
  let v = 100;
  for (let i = 0; i < 20; i++) {
    v += (Math.random() - (trend === 'up' ? 0.4 : 0.6)) * 5;
    points.push(v);
  }
  return points;
}

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

  const today = overview?.todayPnl ?? 0;
  const monthly = overview?.monthlyPnl ?? 0;
  const weekly = overview?.weeklyPnl ?? 0;
  const total = overview?.totalReturn ?? 0;

  // Estimate gross (adding back ~0.05% costs)
  const totalCapital = overview?.totalCapital ?? 1000000;
  const costFactor = 0.0005;

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <h3 className="text-sm font-semibold text-text-primary mb-1">P&L Breakdown</h3>
      <p className="text-2xs text-muted mb-3">Gross and net after transaction costs</p>

      <PnLRow
        label="Today"
        gross={today * (1 + costFactor)}
        net={today}
        grossPct={overview?.todayPnlPct}
        sparkline={mockSparkline(today >= 0 ? 'up' : 'down')}
      />
      <PnLRow
        label="This Week"
        gross={weekly * (1 + costFactor)}
        net={weekly}
        grossPct={overview?.weeklyPnlPct}
        sparkline={mockSparkline(weekly >= 0 ? 'up' : 'down')}
      />
      <PnLRow
        label="This Month"
        gross={monthly * (1 + costFactor)}
        net={monthly}
        grossPct={overview?.monthlyPnlPct}
        sparkline={mockSparkline(monthly >= 0 ? 'up' : 'down')}
      />
      <PnLRow
        label="All Time"
        gross={total + totalCapital * 0.005}
        net={total}
        grossPct={overview?.totalReturnPct}
      />
    </div>
  );
}
