'use client';

import { Header } from '@/components/layout/Header';
import { KillSwitch } from '@/components/common/KillSwitch';
import { StatusBadge } from '@/components/common/StatusBadge';
import { useQuery } from '@tanstack/react-query';
import { healthApi, settingsApi } from '@/lib/api';
import { clsx } from 'clsx';
import { Activity, Wifi, Database, Server, Cpu, Radio, Globe } from 'lucide-react';
import type { ComponentStatus, ComponentStatusType } from '@/types';

const COMPONENT_ICONS: Record<string, React.ReactNode> = {
  API: <Server className="w-4 h-4" />,
  Database: <Database className="w-4 h-4" />,
  Redis: <Cpu className="w-4 h-4" />,
  'Broker Connection': <Radio className="w-4 h-4" />,
  WebSocket: <Wifi className="w-4 h-4" />,
  'Market Data': <Globe className="w-4 h-4" />,
  'Trading Engine': <Activity className="w-4 h-4" />,
};

function statusDotColor(status: ComponentStatusType) {
  return status === 'HEALTHY'
    ? 'bg-profit'
    : status === 'DEGRADED'
    ? 'bg-warning'
    : status === 'DOWN'
    ? 'bg-loss'
    : 'bg-muted';
}

function ComponentCard({ name, status }: { name: string; status: ComponentStatus }) {
  return (
    <div className={clsx(
      'bg-surface border rounded-lg p-4',
      status.status === 'DOWN' ? 'border-loss/40' : status.status === 'DEGRADED' ? 'border-warning/40' : 'border-border',
    )}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={clsx('text-muted', status.status === 'HEALTHY' ? 'text-profit' : status.status === 'DOWN' ? 'text-loss' : 'text-warning')}>
            {COMPONENT_ICONS[name] ?? <Activity className="w-4 h-4" />}
          </div>
          <span className="text-sm font-semibold text-text-primary">{name}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className={clsx('w-2 h-2 rounded-full', statusDotColor(status.status))} />
          <StatusBadge status={status.status} size="sm" />
        </div>
      </div>

      <div className="space-y-1 text-xs">
        <div className="flex justify-between">
          <span className="text-muted">Last Ping</span>
          <span className="font-mono text-text-secondary">
            {new Date(status.lastPing).toLocaleTimeString('en-IN')}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted">Latency</span>
          <span className={clsx('font-mono', status.latencyMs > 200 ? 'text-loss' : status.latencyMs > 100 ? 'text-warning' : 'text-text-secondary')}>
            {status.latencyMs}ms
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted">Uptime</span>
          <span className="font-mono text-text-secondary">{status.uptime.toFixed(2)}%</span>
        </div>
        {status.lastError && (
          <div className="mt-2 p-2 bg-loss/10 border border-loss/20 rounded text-loss text-2xs">
            {status.lastError}
          </div>
        )}
      </div>
    </div>
  );
}

export default function SystemHealthPage() {
  const { data: health, isLoading } = useQuery({
    queryKey: ['system-health', 'detailed'],
    queryFn: () => healthApi.getDetailedHealth(),
    refetchInterval: 15_000,
  });

  const { data: settings } = useQuery({
    queryKey: ['settings'],
    queryFn: () => settingsApi.get(),
    staleTime: 60_000,
  });

  const components = health?.components
    ? [
        { name: 'API', status: health.components.api },
        { name: 'Database', status: health.components.database },
        { name: 'Redis', status: health.components.redis },
        { name: 'Broker Connection', status: health.components.broker },
        { name: 'WebSocket', status: health.components.websocket },
        { name: 'Market Data', status: health.components.marketData },
        { name: 'Trading Engine', status: health.components.tradingEngine },
      ]
    : [];

  // Mock risk events log
  const riskEvents = [
    { time: '10:32:14', type: 'RISK_ALERT', message: 'Daily loss limit 60% consumed', severity: 'MEDIUM' },
    { time: '09:15:02', type: 'SYSTEM', message: 'Market open — intraday strategies activated', severity: 'LOW' },
    { time: '09:00:01', type: 'SYSTEM', message: 'Trading engine started successfully', severity: 'LOW' },
  ];

  return (
    <div className="flex flex-col h-full">
      <Header title="System Health" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Overall Status Banner */}
        <div className={clsx(
          'flex items-center gap-3 p-4 rounded-lg border',
          health?.overall === 'HEALTHY' ? 'bg-profit/10 border-profit/30' : health?.overall === 'DEGRADED' ? 'bg-warning/10 border-warning/30' : 'bg-loss/10 border-loss/30',
        )}>
          <div className={clsx('w-3 h-3 rounded-full', statusDotColor(health?.overall ?? 'UNKNOWN'))} />
          <div>
            <p className="font-semibold text-text-primary">
              System {health?.overall ?? 'Unknown'}
            </p>
            <p className="text-xs text-muted">
              Trading Mode: <span className={health?.tradingMode === 'LIVE' ? 'text-loss font-semibold' : 'text-accent font-semibold'}>{health?.tradingMode ?? '—'}</span>
            </p>
          </div>
        </div>

        {/* Component Grid */}
        <section>
          <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
            <Activity className="w-4 h-4 text-accent" />
            Component Status
          </h3>
          {isLoading ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
              {[...Array(7)].map((_, i) => (
                <div key={i} className="h-32 bg-surface border border-border rounded-lg animate-pulse" />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
              {components.map(({ name, status }) => (
                <ComponentCard key={name} name={name} status={status} />
              ))}
            </div>
          )}
        </section>

        {/* Kill Switch + Reconciliation */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-surface border border-border rounded-lg p-4">
            <h3 className="text-sm font-semibold text-text-primary mb-4">Emergency Kill Switch</h3>
            <KillSwitch
              isActive={health?.killSwitchActive ?? false}
              tradingMode={settings?.tradingMode ?? 'PAPER'}
            />
          </div>

          <div className="bg-surface border border-border rounded-lg p-4">
            <h3 className="text-sm font-semibold text-text-primary mb-4">Reconciliation</h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between items-center">
                <span className="text-muted">Last Run</span>
                <span className="text-text-secondary">
                  {health?.lastReconciliationAt
                    ? new Date(health.lastReconciliationAt).toLocaleString('en-IN')
                    : '—'}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-muted">Result</span>
                <StatusBadge
                  status={
                    health?.reconciliationResult === 'CLEAN' ? 'HEALTHY'
                    : health?.reconciliationResult === 'MISMATCH' ? 'DISABLED'
                    : 'PENDING'
                  }
                  size="sm"
                />
              </div>
              {health?.reconciliationResult === 'MISMATCH' && (
                <div className="p-2.5 bg-loss/10 border border-loss/20 rounded text-xs text-loss">
                  Position mismatch detected between broker and system records. Manual review required.
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Risk Event Log */}
        <section className="bg-surface border border-border rounded-lg">
          <div className="px-4 py-3 border-b border-border">
            <h3 className="text-sm font-semibold text-text-primary">Recent System Events</h3>
          </div>
          <div className="divide-y divide-border">
            {riskEvents.map((event, i) => (
              <div key={i} className="flex items-start gap-3 px-4 py-3 text-xs">
                <span className="font-mono text-muted shrink-0">{event.time}</span>
                <span className={clsx(
                  'shrink-0 px-1.5 py-0.5 rounded text-2xs font-semibold',
                  event.severity === 'MEDIUM' ? 'bg-warning/20 text-warning' : 'bg-muted/20 text-muted',
                )}>
                  {event.type}
                </span>
                <span className="text-text-secondary">{event.message}</span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
