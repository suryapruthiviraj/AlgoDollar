'use client';

import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type { CapitalAllocation } from '@/types';

const COLORS = {
  longterm: '#3b82f6',
  swing: '#6366f1',
  intraday: '#8b5cf6',
  cash: '#374151',
};

const LABELS = {
  longterm: 'Long-Term',
  swing: 'Swing',
  intraday: 'Intraday',
  cash: 'Cash',
};

function formatINR(value: number): string {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0,
    notation: 'compact',
    compactDisplay: 'short',
  }).format(value);
}

interface PieEntry {
  name: string;
  value: number;
  capitalAmount: number;
  riskPct: number;
  color: string;
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: PieEntry }>;
}

function CustomTooltip({ active, payload }: CustomTooltipProps) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-surface-2 border border-border rounded-lg p-3 text-xs shadow-lg">
      <p className="font-semibold text-text-primary mb-2">{d.name}</p>
      <div className="space-y-1">
        <div className="flex justify-between gap-4">
          <span className="text-muted">Capital</span>
          <span className="font-mono text-text-primary">{formatINR(d.capitalAmount)}</span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-muted">Capital %</span>
          <span className="font-mono text-text-primary">{d.value.toFixed(1)}%</span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-muted">Risk %</span>
          <span className="font-mono text-text-primary">{d.riskPct.toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}

interface AllocationPieProps {
  allocation?: CapitalAllocation | null;
  showLegend?: boolean;
}

export function AllocationPie({ allocation, showLegend = true }: AllocationPieProps) {
  const mockAllocation: CapitalAllocation = allocation ?? {
    totalCapital: 1000000,
    longTermCapital: 500000,
    longTermPct: 50,
    swingCapital: 250000,
    swingPct: 25,
    intradayCapital: 150000,
    intradayPct: 15,
    cashBuffer: 100000,
    cashBufferPct: 10,
    longTermRiskPct: 8,
    swingRiskPct: 5,
    intradayRiskPct: 2,
  };

  const pieData: PieEntry[] = [
    {
      name: LABELS.longterm,
      value: mockAllocation.longTermPct,
      capitalAmount: mockAllocation.longTermCapital,
      riskPct: mockAllocation.longTermRiskPct,
      color: COLORS.longterm,
    },
    {
      name: LABELS.swing,
      value: mockAllocation.swingPct,
      capitalAmount: mockAllocation.swingCapital,
      riskPct: mockAllocation.swingRiskPct,
      color: COLORS.swing,
    },
    {
      name: LABELS.intraday,
      value: mockAllocation.intradayPct,
      capitalAmount: mockAllocation.intradayCapital,
      riskPct: mockAllocation.intradayRiskPct,
      color: COLORS.intraday,
    },
    {
      name: LABELS.cash,
      value: mockAllocation.cashBufferPct,
      capitalAmount: mockAllocation.cashBuffer,
      riskPct: 0,
      color: COLORS.cash,
    },
  ];

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={220}>
        <PieChart>
          <Pie
            data={pieData}
            cx="50%"
            cy="50%"
            innerRadius="55%"
            outerRadius="80%"
            paddingAngle={2}
            dataKey="value"
          >
            {pieData.map((entry) => (
              <Cell key={entry.name} fill={entry.color} stroke="transparent" />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
          {showLegend && (
            <Legend
              formatter={(value: string) => (
                <span style={{ color: '#94a3b8', fontSize: 12 }}>{value}</span>
              )}
            />
          )}
        </PieChart>
      </ResponsiveContainer>

      {/* Custom legend with capital + risk % */}
      {showLegend && (
        <div className="grid grid-cols-2 gap-1.5 mt-2">
          {pieData.map((d) => (
            <div key={d.name} className="flex items-center gap-2 text-xs">
              <div className="w-2.5 h-2.5 rounded-sm shrink-0" style={{ backgroundColor: d.color }} />
              <div className="min-w-0">
                <span className="text-text-secondary truncate">{d.name}</span>
                <span className="text-muted ml-1">{d.value.toFixed(0)}%</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
