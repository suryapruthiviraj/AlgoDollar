'use client';

import { Header } from '@/components/layout/Header';
import { StatusBadge } from '@/components/common/StatusBadge';
import { useQuery } from '@tanstack/react-query';
import { marketsApi } from '@/lib/api';
import { clsx } from 'clsx';
import { Globe } from 'lucide-react';
import type { IndexData, SectorData } from '@/types';

function formatINR(v: number) {
  return new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 2 }).format(v);
}

function IndexCard({ index }: { index: IndexData }) {
  const isUp = index.changePct >= 0;
  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <p className="text-xs text-muted mb-1">{index.name}</p>
      <p className="text-xl font-mono font-semibold tabular-nums text-text-primary">
        {index.price.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
      </p>
      <p className={clsx('text-sm font-mono tabular-nums mt-1', isUp ? 'text-profit' : 'text-loss')}>
        {isUp ? '+' : ''}{index.change.toFixed(2)} ({isUp ? '+' : ''}{index.changePct.toFixed(2)}%)
      </p>
      <div className="flex justify-between text-2xs text-muted mt-2">
        <span>H: {index.high.toLocaleString('en-IN', { maximumFractionDigits: 2 })}</span>
        <span>L: {index.low.toLocaleString('en-IN', { maximumFractionDigits: 2 })}</span>
      </div>
    </div>
  );
}

function SectorHeatmap({ sectors }: { sectors: SectorData[] }) {
  const maxAbs = Math.max(...sectors.map((s) => Math.abs(s.return1d)), 1);

  const cellColor = (ret: number) => {
    const intensity = Math.min(Math.abs(ret) / maxAbs, 1);
    if (ret > 0) return `rgba(34, 197, 94, ${0.15 + intensity * 0.55})`;
    if (ret < 0) return `rgba(239, 68, 68, ${0.15 + intensity * 0.55})`;
    return 'rgba(107, 114, 128, 0.15)';
  };

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
      {sectors.map((sector) => (
        <div
          key={sector.sector}
          className="rounded-lg p-3 border border-border cursor-default"
          style={{ backgroundColor: cellColor(sector.return1d) }}
        >
          <p className="text-xs font-semibold text-text-primary truncate">{sector.sector}</p>
          <p className={clsx('text-sm font-mono tabular-nums font-bold mt-1', sector.return1d >= 0 ? 'text-profit' : 'text-loss')}>
            {sector.return1d >= 0 ? '+' : ''}{sector.return1d.toFixed(2)}%
          </p>
          <div className="flex gap-2 text-2xs text-muted mt-1">
            <span>1W: {sector.return1w >= 0 ? '+' : ''}{sector.return1w.toFixed(1)}%</span>
            <span>1M: {sector.return1m >= 0 ? '+' : ''}{sector.return1m.toFixed(1)}%</span>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function MarketsPage() {
  const { data: overview } = useQuery({
    queryKey: ['markets', 'overview'],
    queryFn: () => marketsApi.getOverview(),
    refetchInterval: 30_000,
  });

  const { data: regime } = useQuery({
    queryKey: ['markets', 'regime'],
    queryFn: () => marketsApi.getRegime(),
    refetchInterval: 60_000,
  });

  const { data: sectors } = useQuery({
    queryKey: ['markets', 'sectors'],
    queryFn: () => marketsApi.getSectors(),
    refetchInterval: 60_000,
  });

  const indices = overview?.indices ?? [];
  const sectorData = sectors ?? overview?.sectors ?? [];
  const topGainers = overview?.topGainers ?? [];
  const topLosers = overview?.topLosers ?? [];

  return (
    <div className="flex flex-col h-full">
      <Header title="Markets" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Index Cards */}
        <section>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {indices.length > 0
              ? indices.slice(0, 4).map((idx: IndexData) => <IndexCard key={idx.symbol} index={idx} />)
              : ['NIFTY 50', 'BANK NIFTY', 'NIFTY MIDCAP', 'INDIA VIX'].map((name) => (
                  <div key={name} className="bg-surface border border-border rounded-lg p-4 animate-pulse">
                    <div className="h-3 w-20 bg-border rounded mb-2" />
                    <div className="h-6 w-24 bg-border rounded mb-1" />
                    <div className="h-3 w-16 bg-border rounded" />
                  </div>
                ))}
          </div>
        </section>

        {/* Market Regime + Breadth */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Regime */}
          <div className="bg-surface border border-border rounded-lg p-4">
            <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
              <Globe className="w-4 h-4 text-accent" />
              Market Regime
            </h3>
            {regime ? (
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <StatusBadge
                    status={regime.trend === 'UP' ? 'HEALTHY' : regime.trend === 'DOWN' ? 'DISABLED' : 'REDUCED'}
                    size="md"
                  />
                  <span className="text-sm text-text-primary">{regime.regime.replace(/_/g, ' ')}</span>
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-surface-2 rounded p-2.5">
                    <p className="text-2xs text-muted">Confidence</p>
                    <p className="text-sm font-mono font-semibold tabular-nums">{(regime.confidence * 100).toFixed(0)}%</p>
                  </div>
                  <div className="bg-surface-2 rounded p-2.5">
                    <p className="text-2xs text-muted">VIX</p>
                    <p className="text-sm font-mono font-semibold tabular-nums">{regime.vix.toFixed(1)}</p>
                  </div>
                  <div className="bg-surface-2 rounded p-2.5">
                    <p className="text-2xs text-muted">Breadth</p>
                    <p className="text-sm font-mono font-semibold tabular-nums">{(regime.breadth * 100).toFixed(0)}%</p>
                  </div>
                </div>
              </div>
            ) : (
              <div className="animate-pulse space-y-2">
                <div className="h-6 w-32 bg-border rounded" />
                <div className="h-12 bg-border rounded" />
              </div>
            )}
          </div>

          {/* Market Breadth */}
          <div className="bg-surface border border-border rounded-lg p-4">
            <h3 className="text-sm font-semibold text-text-primary mb-3">Market Breadth</h3>
            {overview ? (
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-muted">Above 50 DMA</span>
                    <span className="font-mono text-text-secondary">{(overview.breadthAbove50dma * 100).toFixed(0)}%</span>
                  </div>
                  <div className="progress-bar">
                    <div className="progress-bar-fill bg-accent" style={{ width: `${overview.breadthAbove50dma * 100}%` }} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-muted">Above 200 DMA</span>
                    <span className="font-mono text-text-secondary">{(overview.breadthAbove200dma * 100).toFixed(0)}%</span>
                  </div>
                  <div className="progress-bar">
                    <div className="progress-bar-fill bg-indigo-500" style={{ width: `${overview.breadthAbove200dma * 100}%` }} />
                  </div>
                </div>
                <div className="flex gap-3 text-xs mt-2">
                  <span className="text-profit">
                    ▲ {overview.advanceDecline.advances} Advances
                  </span>
                  <span className="text-loss">
                    ▼ {overview.advanceDecline.declines} Declines
                  </span>
                  <span className="text-muted">
                    — {overview.advanceDecline.unchanged} Unchanged
                  </span>
                </div>
              </div>
            ) : (
              <div className="animate-pulse space-y-2">
                {[1, 2, 3].map((i) => <div key={i} className="h-6 bg-border rounded" />)}
              </div>
            )}
          </div>
        </section>

        {/* Sector Heatmap */}
        <section className="bg-surface border border-border rounded-lg p-4">
          <h3 className="text-sm font-semibold text-text-primary mb-4">Sector Heatmap (1D Return)</h3>
          {sectorData.length > 0 ? (
            <SectorHeatmap sectors={sectorData} />
          ) : (
            <div className="grid grid-cols-4 gap-2">
              {[...Array(12)].map((_, i) => (
                <div key={i} className="h-20 bg-border rounded animate-pulse" />
              ))}
            </div>
          )}
        </section>

        {/* Top Gainers / Losers */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-surface border border-border rounded-lg p-4">
            <h3 className="text-sm font-semibold text-text-primary mb-3">Top Gainers</h3>
            <div className="space-y-2">
              {topGainers.slice(0, 5).map((g, i) => (
                <div key={g.symbol} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted w-4">{i + 1}</span>
                    <span className="text-sm font-semibold text-text-primary">{g.symbol}</span>
                  </div>
                  <span className="text-sm font-mono text-profit">+{g.changePct.toFixed(2)}%</span>
                </div>
              ))}
              {topGainers.length === 0 && <p className="text-xs text-muted">No data available</p>}
            </div>
          </div>
          <div className="bg-surface border border-border rounded-lg p-4">
            <h3 className="text-sm font-semibold text-text-primary mb-3">Top Losers</h3>
            <div className="space-y-2">
              {topLosers.slice(0, 5).map((l, i) => (
                <div key={l.symbol} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-muted w-4">{i + 1}</span>
                    <span className="text-sm font-semibold text-text-primary">{l.symbol}</span>
                  </div>
                  <span className="text-sm font-mono text-loss">{l.changePct.toFixed(2)}%</span>
                </div>
              ))}
              {topLosers.length === 0 && <p className="text-xs text-muted">No data available</p>}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
