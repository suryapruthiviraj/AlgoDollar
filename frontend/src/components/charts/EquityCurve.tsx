'use client';

import { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useQuery } from '@tanstack/react-query';
import { portfolioApi } from '@/lib/api';
import { clsx } from 'clsx';
import { format, parseISO } from 'date-fns';
import type { EquityCurvePoint } from '@/types';

type Period = '1W' | '1M' | '3M' | '6M' | '1Y' | 'ALL';

const PERIODS: Period[] = ['1W', '1M', '3M', '6M', '1Y', 'ALL'];

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: string;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-surface-2 border border-border rounded-lg p-3 text-xs shadow-lg">
      <p className="text-muted mb-2">{label}</p>
      {payload.map((p) => (
        <div key={p.name} className="flex justify-between gap-6 mb-0.5">
          <span style={{ color: p.color }}>{p.name}</span>
          <span className="font-mono text-text-primary">{p.value.toFixed(2)}</span>
        </div>
      ))}
      {payload.length === 2 && (
        <div className="flex justify-between gap-6 mt-1 pt-1 border-t border-border">
          <span className="text-muted">Excess</span>
          <span
            className={clsx(
              'font-mono',
              payload[0].value >= payload[1].value ? 'text-profit' : 'text-loss',
            )}
          >
            {(payload[0].value - payload[1].value).toFixed(2)}
          </span>
        </div>
      )}
    </div>
  );
}

// Normalize data to 100 at start
function normalizeEquityCurve(data: EquityCurvePoint[]): EquityCurvePoint[] {
  if (data.length === 0) return data;
  const firstPortfolio = data[0].portfolio;
  const firstNifty = data[0].nifty;
  const firstBuyHold = data[0].buyHold ?? firstNifty;

  return data.map((point) => ({
    ...point,
    portfolio: (point.portfolio / firstPortfolio) * 100,
    nifty: (point.nifty / firstNifty) * 100,
    buyHold: point.buyHold ? (point.buyHold / firstBuyHold) * 100 : undefined,
  }));
}

// Generate mock equity curve for display when API is not available
function generateMockCurve(period: Period): EquityCurvePoint[] {
  const days =
    period === '1W' ? 7
    : period === '1M' ? 30
    : period === '3M' ? 90
    : period === '6M' ? 180
    : period === '1Y' ? 365
    : 1000;

  const points: EquityCurvePoint[] = [];
  let portfolio = 100000;
  let nifty = 100000;
  const now = new Date();

  for (let i = days; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    portfolio *= 1 + (Math.random() - 0.47) * 0.015;
    nifty *= 1 + (Math.random() - 0.49) * 0.01;
    points.push({
      date: date.toISOString().split('T')[0],
      portfolio,
      nifty,
    });
  }
  return points;
}

interface EquityCurveProps {
  showBuyHold?: boolean;
  data?: EquityCurvePoint[];
}

export function EquityCurve({ showBuyHold = false, data: externalData }: EquityCurveProps) {
  const [period, setPeriod] = useState<Period>('1M');

  const { data: fetchedData } = useQuery({
    queryKey: ['portfolio', 'performance', period],
    queryFn: async () => {
      const perf = await portfolioApi.getPerformance(period as '1D' | '1W' | '1M' | '3M' | '6M' | '1Y' | '3Y' | 'ALL');
      // API returns equity curve in a real implementation
      return perf;
    },
    staleTime: 60_000,
    enabled: !externalData,
  });

  void fetchedData; // used by API, we use mock for now
  const rawData = externalData ?? generateMockCurve(period);
  const chartData = normalizeEquityCurve(rawData);

  const formatDate = (dateStr: string) => {
    try {
      return format(parseISO(dateStr), period === '1W' ? 'EEE' : period === '1M' ? 'MMM d' : 'MMM yy');
    } catch {
      return dateStr;
    }
  };

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-text-primary">Equity Curve</h3>
        <div className="flex gap-0.5">
          {PERIODS.map((p) => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              className={clsx(
                'px-2 py-0.5 rounded text-xs font-medium transition-colors',
                period === p
                  ? 'bg-accent text-white'
                  : 'text-muted hover:text-text-primary hover:bg-surface-2',
              )}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2028" />
          <XAxis
            dataKey="date"
            tickFormatter={formatDate}
            tick={{ fill: '#6b7280', fontSize: 11 }}
            axisLine={{ stroke: '#1f2028' }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: '#6b7280', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => v.toFixed(0)}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 12, color: '#94a3b8', paddingTop: 8 }}
          />
          <Line
            type="monotone"
            dataKey="portfolio"
            name="Portfolio"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 3 }}
          />
          <Line
            type="monotone"
            dataKey="nifty"
            name="NIFTY 50"
            stroke="#6b7280"
            strokeWidth={1.5}
            strokeDasharray="4 3"
            dot={false}
            activeDot={{ r: 3 }}
          />
          {showBuyHold && (
            <Line
              type="monotone"
              dataKey="buyHold"
              name="Buy & Hold"
              stroke="#f59e0b"
              strokeWidth={1.5}
              strokeDasharray="2 3"
              dot={false}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
