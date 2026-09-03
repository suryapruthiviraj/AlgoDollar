'use client';

import { useQuery } from '@tanstack/react-query';
import { settingsApi } from '@/lib/api';
import { AlertTriangle, Info } from 'lucide-react';
import { clsx } from 'clsx';
import Link from 'next/link';

export function TradingModeBanner() {
  const { data: settings } = useQuery({
    queryKey: ['settings'],
    queryFn: () => settingsApi.get(),
    staleTime: 60_000,
    retry: false,
  });

  if (!settings) return null;

  const isLive = settings.tradingMode === 'LIVE';

  return (
    <div
      className={clsx(
        'flex items-center gap-3 px-4 py-3 rounded-lg border text-sm',
        isLive
          ? 'bg-loss/10 border-loss/30 text-loss'
          : 'bg-accent/10 border-accent/30 text-accent',
      )}
    >
      {isLive ? (
        <AlertTriangle className="w-4 h-4 shrink-0" />
      ) : (
        <Info className="w-4 h-4 shrink-0" />
      )}
      <span className="flex-1">
        {isLive ? (
          <>
            <strong>LIVE TRADING MODE</strong> — Real orders will be placed with your broker. Exercise caution.
          </>
        ) : (
          <>
            <strong>PAPER TRADING MODE</strong> — No real orders will be placed. Safe to experiment.
          </>
        )}
      </span>
      {isLive && (
        <Link
          href="/system-health"
          className="shrink-0 px-3 py-1 border border-loss/40 rounded text-xs font-semibold hover:bg-loss/20 transition-colors"
        >
          Kill Switch
        </Link>
      )}
    </div>
  );
}
