'use client';

import { Bell, Wifi, WifiOff } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { portfolioApi, healthApi, notificationsApi } from '@/lib/api';
import { useRealTimePortfolio } from '@/hooks/usePortfolio';
import { format } from 'date-fns';
import { clsx } from 'clsx';

interface HeaderProps {
  title: string;
}

function formatINR(value: number): string {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0,
  }).format(value);
}

function MarketStatus() {
  const now = new Date();
  // IST offset: UTC+5:30
  const istOffset = 5.5 * 60 * 60 * 1000;
  const ist = new Date(now.getTime() + istOffset - now.getTimezoneOffset() * 60000);
  const hours = ist.getHours();
  const minutes = ist.getMinutes();
  const totalMinutes = hours * 60 + minutes;

  // Market open: 9:15 AM to 3:30 PM IST on weekdays
  const dayOfWeek = ist.getDay(); // 0 = Sunday, 6 = Saturday
  const isWeekday = dayOfWeek >= 1 && dayOfWeek <= 5;
  const isMarketHours = totalMinutes >= 9 * 60 + 15 && totalMinutes < 15 * 60 + 30;
  const isOpen = isWeekday && isMarketHours;

  return (
    <div className="flex items-center gap-2 text-xs">
      <div
        className={clsx(
          'w-1.5 h-1.5 rounded-full',
          isOpen ? 'bg-profit animate-pulse' : 'bg-muted',
        )}
      />
      <span className={isOpen ? 'text-profit' : 'text-muted'}>
        {isOpen ? 'Market Open' : 'Market Closed'}
      </span>
      <span className="text-muted">{format(ist, 'HH:mm')} IST</span>
    </div>
  );
}

function SystemHealthDot() {
  const { data } = useQuery({
    queryKey: ['system-health'],
    queryFn: () => healthApi.getHealth(),
    refetchInterval: 30_000,
    retry: false,
  });

  const status = data?.overall ?? 'UNKNOWN';
  const color =
    status === 'HEALTHY' ? 'bg-profit' : status === 'DEGRADED' ? 'bg-warning' : 'bg-loss';

  return (
    <div
      title={`System: ${status}`}
      className={clsx('w-2 h-2 rounded-full', color)}
    />
  );
}

export function Header({ title }: HeaderProps) {
  const { data: overview } = useQuery({
    queryKey: ['portfolio', 'overview'],
    queryFn: () => portfolioApi.getOverview(),
    refetchInterval: 30_000,
    retry: false,
  });

  const { todayPnl, todayPnlPct, isConnected } = useRealTimePortfolio();

  const { data: notifications } = useQuery({
    queryKey: ['notifications'],
    queryFn: () => notificationsApi.getAll(),
    refetchInterval: 60_000,
    retry: false,
  });

  const unread = notifications?.filter((n) => !n.read).length ?? 0;

  const displayPnl = todayPnl ?? overview?.todayPnl ?? 0;
  const displayPct = todayPnlPct ?? overview?.todayPnlPct ?? 0;
  const totalValue = overview?.currentValue ?? 0;

  return (
    <header className="flex items-center justify-between px-6 py-3 bg-surface border-b border-border min-h-[56px]">
      {/* Left: Title */}
      <h1 className="text-base font-semibold text-text-primary">{title}</h1>

      {/* Right: stats + indicators */}
      <div className="flex items-center gap-5">
        {/* Market status */}
        <MarketStatus />

        {/* Divider */}
        <div className="w-px h-4 bg-border" />

        {/* Portfolio quick stats */}
        <div className="flex items-center gap-4 text-xs">
          <div>
            <span className="text-muted mr-1">Value</span>
            <span className="font-mono font-medium tabular-nums">
              {formatINR(totalValue)}
            </span>
          </div>
          <div>
            <span className="text-muted mr-1">Today</span>
            <span
              className={clsx(
                'font-mono font-medium tabular-nums',
                displayPnl >= 0 ? 'text-profit' : 'text-loss',
              )}
            >
              {displayPnl >= 0 ? '+' : ''}
              {formatINR(displayPnl)} ({displayPct >= 0 ? '+' : ''}
              {displayPct.toFixed(2)}%)
            </span>
          </div>
        </div>

        {/* Divider */}
        <div className="w-px h-4 bg-border" />

        {/* WebSocket indicator */}
        <div title={isConnected ? 'Live data connected' : 'Live data disconnected'}>
          {isConnected ? (
            <Wifi className="w-3.5 h-3.5 text-profit" />
          ) : (
            <WifiOff className="w-3.5 h-3.5 text-muted" />
          )}
        </div>

        {/* System health */}
        <SystemHealthDot />

        {/* Notifications */}
        <button className="relative p-1 text-muted hover:text-text-primary transition-colors">
          <Bell className="w-4 h-4" />
          {unread > 0 && (
            <span className="absolute -top-0.5 -right-0.5 w-3.5 h-3.5 rounded-full bg-loss text-white text-2xs flex items-center justify-center font-bold">
              {unread > 9 ? '9+' : unread}
            </span>
          )}
        </button>
      </div>
    </header>
  );
}
