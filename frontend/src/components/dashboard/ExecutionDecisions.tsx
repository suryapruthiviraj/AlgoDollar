'use client';

import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { executionApi, type ExecutionDecision } from '@/lib/api';

/**
 * Every execution attempt and WHY it ended the way it did.
 *
 * This component exists because an empty trade list is not an answer. It reads
 * identically whether the market was quiet, the feed was dead, the kill switch
 * was engaged, or a sector limit refused the order — four situations that
 * demand four different responses from the operator.
 *
 * So a refusal is shown as
 *
 *     RELIANCE BUY x12   rejected — sector exposure limit
 *
 * with the numbers behind it one click away, and "nothing was attempted" is
 * rendered as its own distinct state rather than as an empty table.
 */

type Filter = 'all' | 'rejected';

const OUTCOME_STYLE: Record<string, { dot: string; text: string }> = {
  SUBMITTED: { dot: 'bg-emerald-500', text: 'text-emerald-400' },
  BLOCKED_RISK: { dot: 'bg-amber-500', text: 'text-amber-400' },
  BLOCKED_KILL_SWITCH: { dot: 'bg-red-500', text: 'text-red-400' },
  BLOCKED_NOT_RECONCILED: { dot: 'bg-red-500', text: 'text-red-400' },
  BLOCKED_ELIGIBILITY: { dot: 'bg-amber-500', text: 'text-amber-400' },
  BLOCKED_MODE: { dot: 'bg-amber-500', text: 'text-amber-400' },
  BLOCKED_DUPLICATE: { dot: 'bg-slate-500', text: 'text-slate-400' },
  AMBIGUOUS: { dot: 'bg-purple-500', text: 'text-purple-400' },
  ERROR: { dot: 'bg-red-500', text: 'text-red-400' },
};

function styleFor(d: ExecutionDecision) {
  // A broker-side rejection carries outcome SUBMITTED — it did reach the venue
  // — so colouring by outcome alone would paint a refused order green.
  if (d.submitted && /rejected|cancelled|expired/i.test(d.headline)) {
    return { dot: 'bg-amber-500', text: 'text-amber-400' };
  }
  return OUTCOME_STYLE[d.outcome] ?? { dot: 'bg-slate-500', text: 'text-slate-400' };
}

function timeOf(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? '—'
    : d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function DecisionRow({ decision }: { decision: ExecutionDecision }) {
  const [open, setOpen] = useState(false);
  const s = styleFor(decision);
  const hasDetail =
    decision.detail || decision.rawReason || decision.failedChecks.length > 0;

  return (
    <li className="border-b border-border last:border-0">
      <button
        type="button"
        onClick={() => hasDetail && setOpen((v) => !v)}
        className={`w-full text-left px-3 py-2.5 flex items-start gap-3 ${
          hasDetail ? 'hover:bg-white/5 cursor-pointer' : 'cursor-default'
        }`}
        aria-expanded={hasDetail ? open : undefined}
      >
        <span className={`mt-1.5 h-2 w-2 rounded-full shrink-0 ${s.dot}`} aria-hidden />
        <span className="flex-1 min-w-0">
          <span className="block text-sm text-text-primary break-words">
            {decision.headline}
          </span>
          <span className="block text-xs text-muted mt-0.5">
            {timeOf(decision.timestamp)}
            {decision.strategy ? ` · ${decision.strategy}` : ''}
            {decision.tradingMode ? ` · ${decision.tradingMode}` : ''}
          </span>
        </span>
        {hasDetail && (
          <span className="text-xs text-muted shrink-0 mt-0.5">{open ? '−' : '+'}</span>
        )}
      </button>

      {open && (
        <div className="px-3 pb-3 pl-8 space-y-1.5 text-xs">
          {decision.detail && (
            <p className="text-text-primary">
              <span className="text-muted">Detail: </span>
              {decision.detail}
            </p>
          )}
          {decision.failedChecks.length > 0 && (
            <p className="text-muted">
              Failed checks:{' '}
              <span className="text-text-primary">
                {decision.failedChecks.join(', ')}
              </span>
            </p>
          )}
          {decision.failedGates.length > 0 && (
            <p className="text-muted">
              Failed gates:{' '}
              <span className="text-text-primary">
                {decision.failedGates.join(', ')}
              </span>
            </p>
          )}
          {decision.reconciliationState && (
            <p className="text-muted">
              Reconciliation:{' '}
              <span className="text-text-primary">{decision.reconciliationState}</span>
            </p>
          )}
          {decision.brokerOrderId && (
            <p className="text-muted">
              Broker order:{' '}
              <span className="text-text-primary font-mono">
                {decision.brokerOrderId}
              </span>
            </p>
          )}
          {decision.rawReason && (
            <p className="text-muted break-words">
              Raw: <span className="font-mono">{decision.rawReason}</span>
            </p>
          )}
        </div>
      )}
    </li>
  );
}

export function ExecutionDecisions({ limit = 15 }: { limit?: number }) {
  const [filter, setFilter] = useState<Filter>('all');

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['execution-decisions', filter, limit],
    queryFn: () =>
      executionApi.getDecisions({ limit, rejectedOnly: filter === 'rejected' }),
    refetchInterval: 15_000,
  });

  return (
    <div className="bg-surface border border-border rounded-lg">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border">
        <div>
          <h3 className="text-sm font-semibold text-text-primary">
            Execution decisions
          </h3>
          <p className="text-xs text-muted mt-0.5">
            Every attempt, and why it ended that way
          </p>
        </div>
        <div className="flex gap-1" role="group" aria-label="Filter decisions">
          {(['all', 'rejected'] as Filter[]).map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => setFilter(f)}
              className={`px-2.5 py-1 text-xs rounded ${
                filter === f
                  ? 'bg-white/10 text-text-primary'
                  : 'text-muted hover:text-text-primary'
              }`}
            >
              {f === 'all' ? 'All' : 'Refused only'}
            </button>
          ))}
        </div>
      </div>

      {data && !isLoading && (
        <div className="px-4 py-2 border-b border-border flex gap-4 text-xs">
          <span className="text-muted">
            Submitted <span className="text-emerald-400">{data.submitted}</span>
          </span>
          <span className="text-muted">
            Refused <span className="text-amber-400">{data.rejected}</span>
          </span>
          <span className="text-muted ml-auto">source: {data.source}</span>
        </div>
      )}

      {isLoading && <p className="px-4 py-6 text-sm text-muted">Loading…</p>}

      {isError && (
        <div className="px-4 py-6 text-sm">
          <p className="text-red-400">The execution trail could not be read.</p>
          <p className="text-xs text-muted mt-1">
            {(error as Error)?.message ?? 'unknown error'}. This is NOT a
            statement that nothing was attempted.
          </p>
        </div>
      )}

      {data && !isLoading && data.entries.length === 0 && (
        <div className="px-4 py-6 text-sm">
          {/*
            The distinction the whole component exists for: an empty list is
            reported as "nothing was attempted", never as "no trades", because
            the latter reads like a decision when it may be a broken feed.
          */}
          <p className="text-text-primary">
            {filter === 'rejected'
              ? 'No orders were refused.'
              : 'No execution attempt has been recorded.'}
          </p>
          <p className="text-xs text-muted mt-1">
            {data.unavailableReason ??
              'Nothing has been attempted — which is not the same as attempts being refused.'}
          </p>
        </div>
      )}

      {data && data.entries.length > 0 && (
        <ul className="max-h-96 overflow-y-auto">
          {data.entries.map((d) => (
            <DecisionRow key={d.auditId} decision={d} />
          ))}
        </ul>
      )}
    </div>
  );
}
