'use client';

import { usePortfolio, useRealTimePortfolio } from '@/hooks/usePortfolio';
import { MetricCard } from '@/components/common/MetricCard';

export function PortfolioOverview() {
  const { overview, isLoading } = usePortfolio();
  const { currentValue, todayPnl, todayPnlPct } = useRealTimePortfolio();

  const liveValue = currentValue ?? overview?.currentValue;
  const livePnl = todayPnl ?? overview?.todayPnl;
  const livePct = todayPnlPct ?? (overview && overview.currentValue
    ? (overview.todayPnl / overview.currentValue) * 100
    : undefined);

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
          value={overview?.invested}
          format="currency"
          subtitle="Market value of positions"
          loading={isLoading}
        />
        <MetricCard
          title="Cash Balance"
          value={overview?.cash}
          format="currency"
          subtitle="Available to deploy"
          loading={isLoading}
        />
        <MetricCard
          title="Current Value"
          value={liveValue}
          format="currency"
          change={liveValue && overview?.invested
            ? liveValue - overview.invested
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
          changePct={overview && overview.currentValue
            ? (overview.monthlyPnl / overview.currentValue) * 100
            : undefined}
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
          value={overview?.drawdownPct}
          format="percent"
          invertColors
          loading={isLoading}
        />
      </div>

      {/* Risk Metrics Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard
          title="Sharpe Ratio"
          value={overview?.sharpe}
          format="number"
          subtitle="Risk-adjusted return"
          loading={isLoading}
        />
        <MetricCard
          title="Sortino Ratio"
          value={overview?.sortino}
          format="number"
          subtitle="Downside risk-adjusted"
          loading={isLoading}
        />
        <MetricCard
          title="Portfolio Vol"
          value={overview?.volatility}
          format="percent"
          subtitle="Annualised"
          loading={isLoading}
        />
        {/* Beta is NOT reported by the backend — computing it needs a
            regression against the index, which nothing here does. The card is
            removed rather than bound to a field that does not exist and would
            render as a confident 0.00. */}
      </div>
    </div>
  );
}
