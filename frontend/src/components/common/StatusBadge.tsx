import { clsx } from 'clsx';

type StatusType =
  | 'HEALTHY'
  | 'REDUCED'
  | 'PAUSED'
  | 'DISABLED'
  | 'PAPER'
  | 'LIVE'
  | 'ACTIVE'
  | 'SHADOW'
  | 'RETIRED'
  | 'TRAINING'
  | 'DEGRADED'
  | 'DOWN'
  | 'UNKNOWN'
  | 'OPEN'
  | 'CLOSED'
  | 'COMPLETE'
  | 'PENDING'
  | 'RUNNING'
  | 'FAILED';

interface StatusBadgeProps {
  status: StatusType | string;
  className?: string;
  size?: 'sm' | 'md';
}

const STATUS_STYLES: Record<string, string> = {
  HEALTHY: 'bg-profit/15 text-profit border-profit/30',
  REDUCED: 'bg-warning/15 text-warning border-warning/30',
  PAUSED: 'bg-warning/15 text-warning border-warning/30',
  DISABLED: 'bg-loss/15 text-loss border-loss/30',
  PAPER: 'bg-accent/15 text-accent border-accent/30',
  LIVE: 'bg-loss/15 text-loss border-loss/30',
  ACTIVE: 'bg-profit/15 text-profit border-profit/30',
  SHADOW: 'bg-muted/15 text-muted border-muted/30',
  RETIRED: 'bg-muted/15 text-muted border-muted/30',
  TRAINING: 'bg-info/15 text-info border-info/30',
  DEGRADED: 'bg-warning/15 text-warning border-warning/30',
  DOWN: 'bg-loss/15 text-loss border-loss/30',
  UNKNOWN: 'bg-muted/15 text-muted border-muted/30',
  OPEN: 'bg-profit/15 text-profit border-profit/30',
  CLOSED: 'bg-muted/15 text-muted border-muted/30',
  COMPLETE: 'bg-profit/15 text-profit border-profit/30',
  PENDING: 'bg-warning/15 text-warning border-warning/30',
  RUNNING: 'bg-info/15 text-info border-info/30',
  FAILED: 'bg-loss/15 text-loss border-loss/30',
};

const PULSE_STATUSES = new Set(['LIVE', 'RUNNING', 'OPEN']);

export function StatusBadge({ status, className, size = 'sm' }: StatusBadgeProps) {
  const styles = STATUS_STYLES[status] ?? 'bg-muted/15 text-muted border-muted/30';
  const shouldPulse = PULSE_STATUSES.has(status);

  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 border rounded font-semibold uppercase tracking-wide',
        size === 'sm' ? 'text-2xs px-1.5 py-0.5' : 'text-xs px-2 py-1',
        styles,
        className,
      )}
    >
      {shouldPulse && (
        <span
          className={clsx(
            'rounded-full',
            size === 'sm' ? 'w-1 h-1' : 'w-1.5 h-1.5',
            status === 'LIVE' ? 'bg-loss animate-pulse-red' : 'bg-current animate-pulse',
          )}
        />
      )}
      {status}
    </span>
  );
}
