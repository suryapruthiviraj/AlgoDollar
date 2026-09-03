'use client';

import { ArrowUp, ArrowDown } from 'lucide-react';
import { clsx } from 'clsx';
import { LineChart, Line, ResponsiveContainer } from 'recharts';

type FormatType = 'currency' | 'percent' | 'number';

interface MetricCardProps {
  title: string;
  value: number | string | null | undefined;
  change?: number;
  changePct?: number;
  format?: FormatType;
  subtitle?: string;
  loading?: boolean;
  sparkline?: number[];
  invertColors?: boolean;
  className?: string;
}

function formatValue(value: number | string | null | undefined, fmt: FormatType): string {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'string') return value;

  switch (fmt) {
    case 'currency':
      return new Intl.NumberFormat('en-IN', {
        style: 'currency',
        currency: 'INR',
        maximumFractionDigits: 0,
      }).format(value);
    case 'percent':
      return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
    case 'number':
      return value.toLocaleString('en-IN', { maximumFractionDigits: 2 });
    default:
      return String(value);
  }
}

function Skeleton() {
  return (
    <div className="animate-pulse">
      <div className="h-3 w-20 bg-border rounded mb-2" />
      <div className="h-6 w-28 bg-border rounded mb-1" />
      <div className="h-2.5 w-16 bg-border rounded" />
    </div>
  );
}

export function MetricCard({
  title,
  value,
  change,
  changePct,
  format = 'number',
  subtitle,
  loading = false,
  sparkline,
  invertColors = false,
  className,
}: MetricCardProps) {
  const changeValue = changePct ?? change ?? 0;
  const isPositive = invertColors ? changeValue < 0 : changeValue >= 0;
  const hasChange = change !== undefined || changePct !== undefined;

  const sparklineData = sparkline?.map((v) => ({ v })) ?? [];

  return (
    <div
      className={clsx(
        'bg-surface border border-border rounded-lg p-4 flex flex-col gap-1',
        className,
      )}
    >
      {loading ? (
        <Skeleton />
      ) : (
        <>
          {/* Title */}
          <p className="text-xs text-muted font-medium uppercase tracking-wide">{title}</p>

          {/* Value row */}
          <div className="flex items-end justify-between gap-2">
            <span className="text-xl font-mono font-semibold tabular-nums text-text-primary">
              {formatValue(value, format)}
            </span>

            {sparkline && sparkline.length > 0 && (
              <div className="w-16 h-8 shrink-0">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={sparklineData}>
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

          {/* Change row */}
          {hasChange && (
            <div
              className={clsx(
                'flex items-center gap-0.5 text-xs font-mono',
                isPositive ? 'text-profit' : 'text-loss',
              )}
            >
              {isPositive ? (
                <ArrowUp className="w-3 h-3" />
              ) : (
                <ArrowDown className="w-3 h-3" />
              )}
              {change !== undefined && (
                <span>{formatValue(Math.abs(change), format)}</span>
              )}
              {changePct !== undefined && (
                <span className="ml-0.5">({Math.abs(changePct).toFixed(2)}%)</span>
              )}
            </div>
          )}

          {/* Subtitle */}
          {subtitle && (
            <p className="text-2xs text-muted mt-0.5">{subtitle}</p>
          )}
        </>
      )}
    </div>
  );
}
