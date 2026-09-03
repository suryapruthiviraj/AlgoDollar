'use client';

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { format, parseISO } from 'date-fns';
import type { DrawdownPoint } from '@/types';

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ value: number }>;
  label?: string;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (!active || !payload?.length) return null;
  const dd = payload[0].value;
  return (
    <div className="bg-surface-2 border border-border rounded-lg p-3 text-xs shadow-lg">
      <p className="text-muted mb-1">{label}</p>
      <p className="font-mono text-loss">{dd.toFixed(2)}%</p>
    </div>
  );
}

function generateMockDrawdown(): DrawdownPoint[] {
  const points: DrawdownPoint[] = [];
  let portfolioValue = 100000;
  let peak = portfolioValue;
  const now = new Date();

  for (let i = 365; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    portfolioValue *= 1 + (Math.random() - 0.49) * 0.012;
    peak = Math.max(peak, portfolioValue);
    const drawdown = ((portfolioValue - peak) / peak) * 100;
    points.push({
      date: date.toISOString().split('T')[0],
      drawdown,
      portfolioValue,
    });
  }
  return points;
}

interface DrawdownChartProps {
  data?: DrawdownPoint[];
}

export function DrawdownChart({ data: externalData }: DrawdownChartProps) {
  const data = externalData ?? generateMockDrawdown();
  const maxDrawdown = Math.min(...data.map((d) => d.drawdown));

  const formatDate = (dateStr: string) => {
    try {
      return format(parseISO(dateStr), 'MMM yy');
    } catch {
      return dateStr;
    }
  };

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-text-primary">Drawdown</h3>
        <span className="text-xs text-muted font-mono">
          Max: <span className="text-loss">{maxDrawdown.toFixed(2)}%</span>
        </span>
      </div>

      <ResponsiveContainer width="100%" height={180}>
        <AreaChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="drawdownGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0.02} />
            </linearGradient>
          </defs>
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
            tickFormatter={(v: number) => `${v.toFixed(0)}%`}
            domain={[Math.floor(maxDrawdown * 1.2), 0]}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine
            y={maxDrawdown}
            stroke="#ef4444"
            strokeDasharray="4 3"
            strokeOpacity={0.6}
            label={{
              value: `Max ${maxDrawdown.toFixed(1)}%`,
              fill: '#ef4444',
              fontSize: 10,
              position: 'insideBottomLeft',
            }}
          />
          <Area
            type="monotone"
            dataKey="drawdown"
            stroke="#ef4444"
            strokeWidth={1.5}
            fill="url(#drawdownGradient)"
            dot={false}
            activeDot={{ r: 3, fill: '#ef4444' }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
