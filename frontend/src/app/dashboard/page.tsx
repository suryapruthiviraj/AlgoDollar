import type { Metadata } from 'next';
import { Header } from '@/components/layout/Header';
import { PortfolioOverview } from '@/components/dashboard/PortfolioOverview';
import { AllocationCard } from '@/components/dashboard/AllocationCard';
import { RiskCard } from '@/components/dashboard/RiskCard';
import { EquityCurve } from '@/components/charts/EquityCurve';
import { PnLCard } from '@/components/dashboard/PnLCard';
import { AllocationPie } from '@/components/charts/AllocationPie';
import { TradingModeBanner } from '@/components/common/TradingModeBanner';
import { ExecutionDecisions } from '@/components/dashboard/ExecutionDecisions';

export const metadata: Metadata = {
  title: 'Dashboard',
};

export default function DashboardPage() {
  return (
    <div className="flex flex-col h-full">
      <Header title="AlgoDollar — Quantitative Portfolio Platform" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Trading Mode Banner */}
        <TradingModeBanner />

        {/* Portfolio Overview */}
        <section>
          <PortfolioOverview />
        </section>

        {/* Allocation + Risk */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <AllocationCard />
          <RiskCard />
        </section>

        {/* Execution decisions.
            Placed above the charts on purpose: when nothing is trading, the
            first question is WHY, and an empty P&L chart cannot answer it. */}
        <section>
          <ExecutionDecisions />
        </section>

        {/* Equity Curve */}
        <section>
          <EquityCurve />
        </section>

        {/* P&L + Allocation Pie */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <PnLCard />
          <div className="bg-surface border border-border rounded-lg p-4">
            <h3 className="text-sm font-semibold text-text-primary mb-4">Capital Allocation</h3>
            <AllocationPie />
          </div>
        </section>
      </div>
    </div>
  );
}
