'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  Briefcase,
  Zap,
  TrendingUp,
  Building2,
  Globe,
  BarChart3,
  ListOrdered,
  Shield,
  Bot,
  Brain,
  FlaskConical,
  Settings,
  Activity,
  FileText,
  ChevronLeft,
  ChevronRight,
  AlertTriangle,
} from 'lucide-react';
import { clsx } from 'clsx';
import { useQuery } from '@tanstack/react-query';
import { settingsApi } from '@/lib/api';

const navItems = [
  { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/portfolio', label: 'Portfolio', icon: Briefcase },
  { href: '/intraday', label: 'Intraday', icon: Zap },
  { href: '/swing', label: 'Swing', icon: TrendingUp },
  { href: '/long-term', label: 'Long-Term', icon: Building2 },
  { href: '/markets', label: 'Markets', icon: Globe },
  { href: '/analytics', label: 'Analytics', icon: BarChart3 },
  { href: '/trades', label: 'Trades', icon: ListOrdered },
  { href: '/risk', label: 'Risk', icon: Shield },
  { href: '/strategies', label: 'Strategies', icon: Bot },
  { href: '/models', label: 'Models', icon: Brain },
  { href: '/research', label: 'Research', icon: FlaskConical },
  { href: '/settings', label: 'Settings', icon: Settings },
  { href: '/system-health', label: 'System Health', icon: Activity },
  { href: '/audit', label: 'Audit Log', icon: FileText },
];

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const pathname = usePathname();

  const { data: settings } = useQuery({
    queryKey: ['settings'],
    queryFn: () => settingsApi.get(),
    refetchInterval: 60_000,
    retry: false,
  });

  const tradingMode = settings?.tradingMode ?? 'PAPER';
  const killSwitchActive = false; // would come from settings or health api

  return (
    <aside
      className={clsx(
        'flex flex-col h-full bg-surface border-r border-border transition-all duration-200 shrink-0',
        collapsed ? 'w-14' : 'w-56',
      )}
    >
      {/* Logo */}
      <div className="flex items-center gap-2 px-3 py-4 border-b border-border min-h-[56px]">
        {!collapsed && (
          <>
            <div className="w-7 h-7 rounded bg-accent flex items-center justify-center text-white text-xs font-bold shrink-0">
              A$
            </div>
            <span className="font-semibold text-sm text-text-primary tracking-wide">AlgoDollar</span>
          </>
        )}
        {collapsed && (
          <div className="w-7 h-7 rounded bg-accent flex items-center justify-center text-white text-xs font-bold mx-auto">
            A$
          </div>
        )}
      </div>

      {/* Trading Mode Badge */}
      {!collapsed && (
        <div className="px-3 py-2 border-b border-border">
          <span
            className={clsx(
              'inline-flex items-center gap-1.5 px-2 py-1 rounded text-xs font-semibold',
              tradingMode === 'LIVE'
                ? 'bg-loss/20 text-loss border border-loss/30'
                : 'bg-accent/20 text-accent border border-accent/30',
            )}
          >
            {tradingMode === 'LIVE' && (
              <span className="w-1.5 h-1.5 rounded-full bg-loss animate-pulse-red" />
            )}
            {tradingMode}
          </span>
        </div>
      )}

      {/* Nav */}
      <nav className="flex-1 py-2 overflow-y-auto">
        {navItems.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || pathname.startsWith(href + '/');
          return (
            <Link
              key={href}
              href={href}
              title={collapsed ? label : undefined}
              className={clsx(
                'flex items-center gap-3 px-3 py-2 mx-1 rounded text-sm transition-colors',
                active
                  ? 'bg-accent/15 text-accent'
                  : 'text-text-secondary hover:bg-surface-2 hover:text-text-primary',
              )}
            >
              <Icon className="w-4 h-4 shrink-0" />
              {!collapsed && <span className="truncate">{label}</span>}
            </Link>
          );
        })}
      </nav>

      {/* Kill Switch */}
      {!collapsed && (
        <div className="px-3 py-3 border-t border-border">
          <button
            className={clsx(
              'w-full flex items-center gap-2 px-3 py-2 rounded text-xs font-semibold transition-colors',
              killSwitchActive
                ? 'bg-loss text-white'
                : 'border border-loss/40 text-loss hover:bg-loss/10',
            )}
            onClick={() => {
              // handled by KillSwitch component on settings/system-health pages
              window.location.href = '/system-health';
            }}
          >
            <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
            {killSwitchActive ? 'KILL SWITCH ACTIVE' : 'Kill Switch'}
          </button>
        </div>
      )}

      {/* Collapse toggle */}
      <button
        className="flex items-center justify-center py-3 border-t border-border text-muted hover:text-text-primary transition-colors"
        onClick={() => setCollapsed((c) => !c)}
        aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
      </button>
    </aside>
  );
}
