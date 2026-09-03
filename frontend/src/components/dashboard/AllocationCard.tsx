'use client';

import { useState } from 'react';
import { useAllocation } from '@/hooks/usePortfolio';
import { useQuery } from '@tanstack/react-query';
import { settingsApi } from '@/lib/api';
import { AllocationPie } from '@/components/charts/AllocationPie';
import { clsx } from 'clsx';
import { Calculator, CheckCircle, AlertTriangle } from 'lucide-react';

function formatINRInput(value: string): string {
  const num = parseInt(value.replace(/[^0-9]/g, ''), 10);
  if (isNaN(num)) return '';
  return new Intl.NumberFormat('en-IN').format(num);
}

export function AllocationCard() {
  const [rawInput, setRawInput] = useState('');
  const [displayInput, setDisplayInput] = useState('');
  const { calculate, execute, pendingAllocation } = useAllocation();

  const { data: settings } = useQuery({
    queryKey: ['settings'],
    queryFn: () => settingsApi.get(),
    staleTime: 60_000,
    retry: false,
  });

  const isLive = settings?.tradingMode === 'LIVE';
  const killSwitchActive = false;

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value.replace(/[^0-9]/g, '');
    setRawInput(raw);
    setDisplayInput(raw ? formatINRInput(raw) : '');
  };

  const handleCalculate = () => {
    const amount = parseInt(rawInput, 10);
    if (!isNaN(amount) && amount > 0) {
      calculate.mutate(amount);
    }
  };

  const handleExecute = () => {
    if (pendingAllocation?.id) {
      execute.mutate(pendingAllocation.id);
    }
  };

  return (
    <div className="bg-surface border border-border rounded-lg p-4 flex flex-col gap-4">
      <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
        <Calculator className="w-4 h-4 text-accent" />
        Monthly Contribution Allocator
      </h3>

      {/* Input */}
      <div>
        <label className="block text-xs text-muted mb-1.5">Monthly Contribution (₹)</label>
        <div className="flex gap-2">
          <div className="relative flex-1">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted text-sm">₹</span>
            <input
              type="text"
              inputMode="numeric"
              value={displayInput}
              onChange={handleInputChange}
              placeholder="10,000"
              className="w-full bg-surface-2 border border-border rounded px-3 py-2 pl-7 text-sm font-mono text-text-primary placeholder:text-muted focus:border-accent focus:outline-none"
            />
          </div>
          <button
            onClick={handleCalculate}
            disabled={!rawInput || calculate.isPending}
            className="px-4 py-2 bg-accent text-white rounded text-sm font-medium hover:bg-accent-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
          >
            {calculate.isPending ? 'Calculating...' : 'Calculate'}
          </button>
        </div>
      </div>

      {/* Calculation Error */}
      {calculate.isError && (
        <div className="flex items-center gap-2 p-3 bg-loss/10 border border-loss/20 rounded text-xs text-loss">
          <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
          Failed to calculate allocation. Please try again.
        </div>
      )}

      {/* Results */}
      {pendingAllocation && (
        <div className="space-y-3 animate-fade-in">
          <div className="h-px bg-border" />

          {/* Allocation Pie */}
          <AllocationPie allocation={pendingAllocation.recommendedAllocation} />

          {/* Allocation breakdown */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            {[
              {
                label: 'Long-Term',
                amount: pendingAllocation.recommendedAllocation.longTermCapital,
                pct: pendingAllocation.recommendedAllocation.longTermPct,
                color: 'text-accent',
              },
              {
                label: 'Swing',
                amount: pendingAllocation.recommendedAllocation.swingCapital,
                pct: pendingAllocation.recommendedAllocation.swingPct,
                color: 'text-indigo-400',
              },
              {
                label: 'Intraday',
                amount: pendingAllocation.recommendedAllocation.intradayCapital,
                pct: pendingAllocation.recommendedAllocation.intradayPct,
                color: 'text-purple-400',
              },
              {
                label: 'Cash Buffer',
                amount: pendingAllocation.recommendedAllocation.cashBuffer,
                pct: pendingAllocation.recommendedAllocation.cashBufferPct,
                color: 'text-muted',
              },
            ].map((item) => (
              <div key={item.label} className="flex justify-between items-center bg-surface-2 rounded px-2.5 py-1.5">
                <span className={clsx('font-medium', item.color)}>{item.label}</span>
                <div className="text-right">
                  <div className="font-mono text-text-primary">
                    ₹{new Intl.NumberFormat('en-IN').format(item.amount)}
                  </div>
                  <div className="text-muted">{item.pct.toFixed(0)}%</div>
                </div>
              </div>
            ))}
          </div>

          {/* Explanation */}
          {pendingAllocation.explanation && (
            <p className="text-xs text-text-secondary leading-relaxed">
              {pendingAllocation.explanation}
            </p>
          )}

          {/* Execute button */}
          <button
            onClick={handleExecute}
            disabled={execute.isPending || killSwitchActive}
            className={clsx(
              'w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded text-sm font-semibold transition-colors',
              killSwitchActive
                ? 'bg-border text-muted cursor-not-allowed'
                : isLive
                ? 'bg-loss text-white hover:bg-red-600'
                : 'bg-profit/20 text-profit border border-profit/30 hover:bg-profit/30',
            )}
          >
            <CheckCircle className="w-4 h-4" />
            {execute.isPending
              ? 'Executing...'
              : killSwitchActive
              ? 'Kill Switch Active'
              : isLive
              ? 'Execute Allocation (LIVE)'
              : 'Execute Allocation (Paper)'}
          </button>

          {isLive && !killSwitchActive && (
            <p className="text-xs text-loss text-center">
              This will place real orders with your broker.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
