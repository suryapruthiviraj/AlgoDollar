'use client';

import { useState } from 'react';
import { Header } from '@/components/layout/Header';
import { useQuery } from '@tanstack/react-query';
import { auditApi, type AuditFilters } from '@/lib/api';
import { clsx } from 'clsx';
import { FileText, Download, ChevronDown, ChevronUp } from 'lucide-react';
import type { AuditLog } from '@/types';

function exportAuditCSV(logs: AuditLog[]) {
  const headers = ['Timestamp', 'Action', 'Entity Type', 'Entity ID', 'User', 'IP', 'Details'];
  const rows = logs.map((l) => [
    l.timestamp,
    l.action,
    l.entityType,
    l.entityId,
    l.user,
    l.ip ?? '',
    l.details ?? '',
  ]);
  const csv = [headers, ...rows].map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `algodollar-audit-${new Date().toISOString().split('T')[0]}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function JsonDiff({ before, after }: { before?: Record<string, unknown>; after?: Record<string, unknown> }) {
  if (!before && !after) return null;

  const allKeys = new Set([...Object.keys(before ?? {}), ...Object.keys(after ?? {})]);

  return (
    <div className="grid grid-cols-2 gap-3 text-2xs font-mono">
      <div>
        <p className="text-muted mb-1 text-xs">Before</p>
        <div className="bg-surface rounded p-2 space-y-0.5">
          {before
            ? Array.from(allKeys).map((key) => (
                <div key={key} className={clsx(after?.[key] !== before?.[key] ? 'text-loss' : 'text-text-secondary')}>
                  <span className="text-muted">{key}: </span>
                  {JSON.stringify(before[key])}
                </div>
              ))
            : <span className="text-muted">—</span>}
        </div>
      </div>
      <div>
        <p className="text-muted mb-1 text-xs">After</p>
        <div className="bg-surface rounded p-2 space-y-0.5">
          {after
            ? Array.from(allKeys).map((key) => (
                <div key={key} className={clsx(before?.[key] !== after?.[key] ? 'text-profit' : 'text-text-secondary')}>
                  <span className="text-muted">{key}: </span>
                  {JSON.stringify(after[key])}
                </div>
              ))
            : <span className="text-muted">—</span>}
        </div>
      </div>
    </div>
  );
}

function AuditRow({ log }: { log: AuditLog }) {
  const [expanded, setExpanded] = useState(false);
  const hasDiff = log.before || log.after;

  return (
    <>
      <tr className="hover:bg-surface-2 transition-colors">
        <td className="font-mono text-2xs text-muted">
          {new Date(log.timestamp).toLocaleString('en-IN', { hour12: false })}
        </td>
        <td className="font-semibold text-xs text-text-primary">{log.action}</td>
        <td className="text-xs text-text-secondary">{log.entityType}</td>
        <td className="font-mono text-xs text-muted">{log.entityId}</td>
        <td className="text-xs text-text-secondary">{log.user}</td>
        <td className="text-xs text-muted font-mono">{log.ip ?? '—'}</td>
        <td className="text-xs text-text-secondary max-w-[180px] truncate">{log.details ?? '—'}</td>
        <td>
          {hasDiff && (
            <button
              onClick={() => setExpanded((e) => !e)}
              className="text-xs text-accent hover:text-accent-hover flex items-center gap-0.5"
            >
              Diff
              {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>
          )}
        </td>
      </tr>
      {expanded && hasDiff && (
        <tr className="bg-surface-2">
          <td colSpan={8} className="px-4 py-3">
            <JsonDiff before={log.before} after={log.after} />
          </td>
        </tr>
      )}
    </>
  );
}

const ACTION_TYPES = [
  'ALL', 'SETTINGS_CHANGE', 'KILL_SWITCH', 'STRATEGY_OVERRIDE',
  'ORDER_PLACED', 'ORDER_CANCELLED', 'ALLOCATION_EXECUTED', 'LOGIN', 'LOGOUT',
];

const ENTITY_TYPES = ['ALL', 'settings', 'strategy', 'order', 'allocation', 'position', 'user'];

export default function AuditPage() {
  const [filters, setFilters] = useState<AuditFilters>({});

  const { data: logsData, isLoading } = useQuery({
    queryKey: ['audit', filters],
    queryFn: () => auditApi.getLogs(filters),
    staleTime: 30_000,
  });

  const logs = logsData?.logs ?? [];
  const total = logsData?.total ?? 0;

  return (
    <div className="flex flex-col h-full">
      <Header title="Audit Log" />

      <div className="flex-1 overflow-y-auto p-6 space-y-5">
        {/* Filters */}
        <div className="bg-surface border border-border rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <FileText className="w-4 h-4 text-accent" />
            <h3 className="text-sm font-semibold text-text-primary">Filters</h3>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            <div>
              <label className="block text-2xs text-muted mb-1">From</label>
              <input
                type="date"
                value={filters.startDate ?? ''}
                onChange={(e) => setFilters((f) => ({ ...f, startDate: e.target.value || undefined }))}
                className="w-full bg-surface-2 border border-border rounded px-2 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-2xs text-muted mb-1">To</label>
              <input
                type="date"
                value={filters.endDate ?? ''}
                onChange={(e) => setFilters((f) => ({ ...f, endDate: e.target.value || undefined }))}
                className="w-full bg-surface-2 border border-border rounded px-2 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-2xs text-muted mb-1">Action Type</label>
              <select
                value={filters.action ?? 'ALL'}
                onChange={(e) => setFilters((f) => ({ ...f, action: e.target.value === 'ALL' ? undefined : e.target.value }))}
                className="w-full bg-surface-2 border border-border rounded px-2 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
              >
                {ACTION_TYPES.map((a) => <option key={a} value={a}>{a}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-2xs text-muted mb-1">Entity Type</label>
              <select
                value={filters.entityType ?? 'ALL'}
                onChange={(e) => setFilters((f) => ({ ...f, entityType: e.target.value === 'ALL' ? undefined : e.target.value }))}
                className="w-full bg-surface-2 border border-border rounded px-2 py-1.5 text-xs text-text-primary focus:border-accent focus:outline-none"
              >
                {ENTITY_TYPES.map((e) => <option key={e} value={e}>{e}</option>)}
              </select>
            </div>
            <div className="flex items-end gap-2">
              <button
                onClick={() => setFilters({})}
                className="flex-1 px-3 py-1.5 border border-border rounded text-xs text-muted hover:text-text-primary hover:border-muted transition-colors"
              >
                Clear
              </button>
              <button
                onClick={() => exportAuditCSV(logs)}
                disabled={logs.length === 0}
                className="flex items-center gap-1 px-3 py-1.5 bg-surface-2 border border-border rounded text-xs text-text-secondary hover:text-text-primary disabled:opacity-50 transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                CSV
              </button>
            </div>
          </div>
        </div>

        {/* Results count */}
        {total > 0 && (
          <p className="text-xs text-muted px-1">{total.toLocaleString('en-IN')} entries found</p>
        )}

        {/* Audit Table */}
        <section className="bg-surface border border-border rounded-lg overflow-hidden">
          {isLoading ? (
            <div className="p-8 text-center text-muted text-sm animate-pulse">Loading audit logs...</div>
          ) : logs.length === 0 ? (
            <div className="py-12 text-center text-muted text-sm">
              No audit log entries found for the selected filters.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Action</th>
                    <th>Entity Type</th>
                    <th>Entity ID</th>
                    <th>User</th>
                    <th>IP</th>
                    <th>Details</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {logs.map((log: AuditLog) => (
                    <AuditRow key={log.id} log={log} />
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
