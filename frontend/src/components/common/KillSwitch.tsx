'use client';

import { useState } from 'react';
import { AlertTriangle, X } from 'lucide-react';
import { clsx } from 'clsx';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { settingsApi } from '@/lib/api';

interface KillSwitchProps {
  isActive: boolean;
  tradingMode?: 'PAPER' | 'LIVE';
}

export function KillSwitch({ isActive, tradingMode = 'PAPER' }: KillSwitchProps) {
  const [showModal, setShowModal] = useState(false);
  const [confirmText, setConfirmText] = useState('');
  const queryClient = useQueryClient();

  const toggle = useMutation({
    mutationFn: (active: boolean) => settingsApi.toggleKillSwitch(active),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] });
      queryClient.invalidateQueries({ queryKey: ['system-health'] });
      setShowModal(false);
      setConfirmText('');
    },
  });

  const isLive = tradingMode === 'LIVE';
  const requiresConfirm = isLive && !isActive;
  const canActivate = !requiresConfirm || confirmText === 'CONFIRM';

  return (
    <>
      {/* Kill Switch Button */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center gap-3">
          <div>
            <p className="text-sm font-semibold text-text-primary">Emergency Kill Switch</p>
            <p className="text-xs text-muted">
              {isActive
                ? 'Trading is halted. All pending orders cancelled.'
                : 'Activating will stop all trading and cancel pending orders.'}
            </p>
          </div>
          <div
            className={clsx(
              'px-2 py-1 rounded text-xs font-semibold',
              isActive
                ? 'bg-loss/20 text-loss border border-loss/30'
                : 'bg-profit/20 text-profit border border-profit/30',
            )}
          >
            {isActive ? 'ACTIVE' : 'INACTIVE'}
          </div>
        </div>

        <button
          onClick={() => setShowModal(true)}
          disabled={toggle.isPending}
          className={clsx(
            'flex items-center gap-2 px-4 py-2.5 rounded font-semibold text-sm transition-colors',
            isActive
              ? 'bg-surface-2 border border-border text-text-secondary hover:text-text-primary'
              : 'bg-loss text-white hover:bg-red-600 active:bg-red-700',
          )}
        >
          <AlertTriangle className="w-4 h-4" />
          {isActive ? 'Deactivate Kill Switch' : 'Activate Kill Switch'}
        </button>
      </div>

      {/* Confirmation Modal */}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-surface border border-border rounded-xl p-6 w-full max-w-md mx-4 animate-fade-in">
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-loss/20 flex items-center justify-center">
                  <AlertTriangle className="w-5 h-5 text-loss" />
                </div>
                <div>
                  <h3 className="font-semibold text-text-primary">
                    {isActive ? 'Deactivate Kill Switch?' : 'Activate Kill Switch?'}
                  </h3>
                  <p className="text-xs text-muted">This action affects live trading</p>
                </div>
              </div>
              <button
                onClick={() => { setShowModal(false); setConfirmText(''); }}
                className="text-muted hover:text-text-primary"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Body */}
            <div className="bg-loss/10 border border-loss/20 rounded-lg p-3 mb-4">
              <ul className="text-sm text-text-secondary space-y-1">
                <li>• All active strategies will be halted immediately</li>
                <li>• All pending orders will be cancelled</li>
                <li>• No new signals will be processed</li>
                <li>• Open positions will NOT be automatically closed</li>
              </ul>
            </div>

            {/* Live mode confirmation */}
            {requiresConfirm && (
              <div className="mb-4">
                <p className="text-xs text-muted mb-2">
                  Type <span className="font-mono font-bold text-text-primary">CONFIRM</span> to activate:
                </p>
                <input
                  type="text"
                  value={confirmText}
                  onChange={(e) => setConfirmText(e.target.value)}
                  placeholder="Type CONFIRM"
                  className="w-full bg-surface-2 border border-border rounded px-3 py-2 text-sm font-mono text-text-primary focus:border-accent focus:outline-none"
                />
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-2">
              <button
                onClick={() => { setShowModal(false); setConfirmText(''); }}
                className="flex-1 px-4 py-2 border border-border rounded text-sm text-text-secondary hover:text-text-primary hover:border-muted transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => toggle.mutate(!isActive)}
                disabled={!canActivate || toggle.isPending}
                className={clsx(
                  'flex-1 px-4 py-2 rounded text-sm font-semibold transition-colors',
                  isActive
                    ? 'bg-profit text-white hover:bg-green-600 disabled:opacity-50'
                    : 'bg-loss text-white hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed',
                )}
              >
                {toggle.isPending
                  ? 'Processing...'
                  : isActive
                  ? 'Deactivate'
                  : 'Activate Kill Switch'}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
