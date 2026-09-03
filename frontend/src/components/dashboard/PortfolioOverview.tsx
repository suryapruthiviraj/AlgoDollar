'use client';

import { usePortfolio, useRealTimePortfolio } from '@/hooks/usePortfolio';
import { MetricCard } from '@/components/common/MetricCard';

export function PortfolioOverview() {
  const { overview, isLoading } = usePortfolio();
  const { currentValue, todayPnl, todayPnlPct } = useRealTimePortfolio();

  const liveValue = currentValue ?? overview?.currentValue;
  const livePnl = todayPnl ?? overview?.todayPnl;
  const livePct = todayPnlPct ?? overview?.todayPnlPct;

  return (
    <div className="space-y-3">
      {/* Capital Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard
          title="Total Capital"
          value={overview?.totalCapital}
          format="currency"
          subtitle="Deployed + Cash"
          loading={isLoading}
        />
        <MetricCard
          title="Invested"
          value={overview?.investedCapital}
          format="currency"
          subtitle="Market value of positions"
          loading={isLoading}
        />
        <MetricCard
          title="Cash Balance"
          value={overview?.cashBalance}
          format="currency"
          subtitle="Available to deploy"
          loading={isLoading}
        />
        <MetricCard
          title="Current Value"
          value={liveValue}
          format="currency"
          change={liveValue && overview?.investedCapital
            ? liveValue - overview.investedCapital
            : undefined}
          subtitle="Live market value"
          loading={isLoading}
        />
      </div>

      {/* P&L Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard
          title="Today P&L"
          value={livePnl}
          changePct={livePct}
          format="currency"
          loading={isLoading}
        />
        <MetricCard
          title="Monthly P&L"
          value={overview?.monthlyPnl}
          changePct={overview?.monthlyPnlPct}
          format="currency"
          loading={isLoading}
        />
        <MetricCard
          title="Total Return"
          value={overview?.totalReturnPct}
          format="percent"
          loading={isLoading}
        />
        <MetricCard
          title="Max Drawdown"
          value={overview?.maxDrawdown}
          format="percent"
          invertColors
          loading={isLoading}
        />
      </div>

      {/* Risk Metrics Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard
          title="Sharpe Ratio"
          value={overview?.sharpeRatio}
          format="number"
          subtitle="Risk-adjusted return"
          loading={isLoading}
        />
        <MetricCard
          title="Sortino Ratio"
          value={overview?.sortinoRatio}
          format="number"
          subtitle="Downside risk-adjusted"
          loading={isLoading}
        />
        <MetricCard
          title="Portfolio Vol"
          value={overview?.portfolioVol}
          format="percent"
          subtitle="Annualised"
          loading={isLoading}
        />
        <MetricCard
          title="Beta"
          value={overview?.beta}
          format="number"
          subtitle="vs NIFTY 50"
          loading={isLoading}
        />
      </div>
    </div>
  );
}
