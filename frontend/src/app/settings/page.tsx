'use client';

import { useState, useEffect } from 'react';
import { Header } from '@/components/layout/Header';
import { KillSwitch } from '@/components/common/KillSwitch';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { settingsApi } from '@/lib/api';
import { clsx } from 'clsx';
import { Settings, ChevronDown, ChevronUp, AlertTriangle, Save } from 'lucide-react';
import type { UserSettings } from '@/types';

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className={clsx(
        'relative inline-flex h-5 w-9 rounded-full transition-colors',
        checked ? 'bg-accent' : 'bg-border',
      )}
    >
      <span
        className={clsx(
          'inline-block h-4 w-4 rounded-full bg-white shadow transition-transform mt-0.5',
          checked ? 'translate-x-4.5 ml-0.5' : 'ml-0.5',
        )}
      />
    </button>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-surface border border-border rounded-lg p-5 space-y-4">
      <h3 className="text-sm font-semibold text-text-primary">{title}</h3>
      {children}
    </div>
  );
}

function FieldRow({ label, description, children }: { label: string; description?: string; children: React.ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4">
      <div className="flex-1 min-w-0">
        <p className="text-sm text-text-primary">{label}</p>
        {description && <p className="text-xs text-muted mt-0.5">{description}</p>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

function NumberInput({ value, onChange, min, max, step = 1, prefix, suffix }: {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  prefix?: string;
  suffix?: string;
}) {
  return (
    <div className="flex items-center gap-1">
      {prefix && <span className="text-muted text-sm">{prefix}</span>}
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        min={min}
        max={max}
        step={step}
        className="w-24 bg-surface-2 border border-border rounded px-2 py-1 text-sm font-mono text-right text-text-primary focus:border-accent focus:outline-none"
      />
      {suffix && <span className="text-muted text-sm">{suffix}</span>}
    </div>
  );
}

const DEFAULT_SETTINGS: UserSettings = {
  tradingMode: 'PAPER',
  monthlyCapital: 50000,
  riskTolerance: 'MEDIUM',
  strategies: { intraday: true, swing: true, longterm: true },
  autoExecution: false,
  riskLimits: {
    maxPortfolioDrawdown: 15,
    maxDailyLoss: 5000,
    maxWeeklyLoss: 15000,
    maxMonthlyLoss: 40000,
    maxPositionSize: 10,
    maxSectorConcentration: 30,
    maxCorrelation: 0.8,
    intradayMaxPositions: 3,
    swingMaxPositions: 8,
    longtermMaxPositions: 20,
    rebalancingThreshold: 5,
  },
  sectorLimits: {},
  rebalancingThreshold: 5,
  notifications: {
    riskAlerts: true,
    tradeExecutions: true,
    dailySummary: true,
    weeklyReport: true,
  },
};

export default function SettingsPage() {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);
  const [localSettings, setLocalSettings] = useState<UserSettings>(DEFAULT_SETTINGS);
  const queryClient = useQueryClient();

  const { data: settings, isLoading } = useQuery({
    queryKey: ['settings'],
    queryFn: () => settingsApi.get(),
    staleTime: 60_000,
  });

  useEffect(() => {
    if (settings) setLocalSettings(settings);
  }, [settings]);

  const save = useMutation({
    mutationFn: (s: Partial<UserSettings>) => settingsApi.update(s),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] });
      setShowSaveModal(false);
    },
  });

  const set = <K extends keyof UserSettings>(key: K, value: UserSettings[K]) => {
    setLocalSettings((s) => ({ ...s, [key]: value }));
  };

  const setRisk = <K extends keyof UserSettings['riskLimits']>(key: K, value: number) => {
    setLocalSettings((s) => ({ ...s, riskLimits: { ...s.riskLimits, [key]: value } }));
  };

  if (isLoading) {
    return (
      <div className="flex flex-col h-full">
        <Header title="Settings" />
        <div className="flex-1 p-6 space-y-4 animate-pulse">
          {[1, 2, 3, 4].map((i) => <div key={i} className="h-32 bg-surface border border-border rounded-lg" />)}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <Header title="Settings" />

      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {/* 1. Trading Mode */}
        <Section title="Trading Mode">
          <div className={clsx(
            'p-4 rounded-lg border',
            localSettings.tradingMode === 'LIVE'
              ? 'bg-loss/10 border-loss/30'
              : 'bg-accent/10 border-accent/30',
          )}>
            <div className="flex items-center justify-between mb-3">
              <div>
                <p className="font-semibold text-text-primary">
                  {localSettings.tradingMode === 'LIVE' ? 'LIVE TRADING' : 'PAPER TRADING'}
                </p>
                <p className="text-xs text-muted mt-0.5">
                  {localSettings.tradingMode === 'LIVE'
                    ? 'Real money. Real consequences. Double-check everything.'
                    : 'Simulated trading. No real orders placed.'}
                </p>
              </div>
              <div className="flex gap-1">
                {['PAPER', 'LIVE'].map((mode) => (
                  <button
                    key={mode}
                    onClick={() => set('tradingMode', mode as 'PAPER' | 'LIVE')}
                    className={clsx(
                      'px-3 py-1.5 rounded text-xs font-semibold transition-colors',
                      localSettings.tradingMode === mode
                        ? mode === 'LIVE' ? 'bg-loss text-white' : 'bg-accent text-white'
                        : 'bg-surface-2 border border-border text-muted hover:text-text-primary',
                    )}
                  >
                    {mode}
                  </button>
                ))}
              </div>
            </div>
            {localSettings.tradingMode === 'LIVE' && (
              <div className="flex items-center gap-2 text-xs text-loss">
                <AlertTriangle className="w-3.5 h-3.5" />
                Switching to LIVE will place real orders. Ensure broker credentials are configured.
              </div>
            )}
          </div>
        </Section>

        {/* Kill Switch */}
        <Section title="Kill Switch">
          <KillSwitch isActive={false} tradingMode={localSettings.tradingMode} />
        </Section>

        {/* 2. Monthly Capital */}
        <Section title="Monthly Capital">
          <FieldRow label="Monthly Contribution" description="Amount added to portfolio each month for allocation">
            <NumberInput
              value={localSettings.monthlyCapital}
              onChange={(v) => set('monthlyCapital', v)}
              min={0}
              step={5000}
              prefix="₹"
            />
          </FieldRow>
        </Section>

        {/* 3. Risk Tolerance */}
        <Section title="Risk Tolerance">
          <div className="flex gap-2">
            {(['LOW', 'MEDIUM', 'HIGH'] as const).map((tol) => (
              <button
                key={tol}
                onClick={() => set('riskTolerance', tol)}
                className={clsx(
                  'flex-1 py-2.5 rounded border text-sm font-semibold transition-colors',
                  localSettings.riskTolerance === tol
                    ? tol === 'HIGH' ? 'bg-loss/20 border-loss/40 text-loss'
                      : tol === 'LOW' ? 'bg-profit/20 border-profit/40 text-profit'
                      : 'bg-accent/20 border-accent/40 text-accent'
                    : 'bg-surface-2 border-border text-muted hover:text-text-primary',
                )}
              >
                {tol}
              </button>
            ))}
          </div>
          <p className="text-xs text-muted">
            {localSettings.riskTolerance === 'LOW' && 'Conservative allocation. Lower position sizes. Wider stops.'}
            {localSettings.riskTolerance === 'MEDIUM' && 'Balanced approach. Moderate position sizing. Default parameters.'}
            {localSettings.riskTolerance === 'HIGH' && 'Aggressive allocation. Larger positions. Tighter stops. Higher drawdown tolerance.'}
          </p>
        </Section>

        {/* 4. Strategy Toggles */}
        <Section title="Strategies">
          {(['intraday', 'swing', 'longterm'] as const).map((strategy) => (
            <FieldRow
              key={strategy}
              label={strategy === 'longterm' ? 'Long-Term' : strategy === 'swing' ? 'Swing' : 'Intraday'}
              description={strategy === 'intraday' ? 'Day trading — all positions closed by EOD'
                : strategy === 'swing' ? 'Multi-day trades — 2-20 day holds'
                : 'Fundamental investing — weeks to months'}
            >
              <Toggle
                checked={localSettings.strategies[strategy]}
                onChange={(v) => set('strategies', { ...localSettings.strategies, [strategy]: v })}
              />
            </FieldRow>
          ))}
          <FieldRow label="Auto-Execution" description="Automatically execute signals without manual approval">
            <Toggle
              checked={localSettings.autoExecution}
              onChange={(v) => set('autoExecution', v)}
            />
          </FieldRow>
        </Section>

        {/* 5. Risk Limits */}
        <Section title="Risk Limits">
          <FieldRow label="Max Portfolio Drawdown" description="Halt all trading when portfolio drawdown exceeds this">
            <NumberInput value={localSettings.riskLimits.maxPortfolioDrawdown} onChange={(v) => setRisk('maxPortfolioDrawdown', v)} min={1} max={50} step={0.5} suffix="%" />
          </FieldRow>
          <FieldRow label="Max Daily Loss" description="Stop intraday trading when daily loss exceeds this">
            <NumberInput value={localSettings.riskLimits.maxDailyLoss} onChange={(v) => setRisk('maxDailyLoss', v)} min={1000} step={500} prefix="₹" />
          </FieldRow>
          <FieldRow label="Max Weekly Loss">
            <NumberInput value={localSettings.riskLimits.maxWeeklyLoss} onChange={(v) => setRisk('maxWeeklyLoss', v)} min={1000} step={1000} prefix="₹" />
          </FieldRow>
          <FieldRow label="Max Monthly Loss">
            <NumberInput value={localSettings.riskLimits.maxMonthlyLoss} onChange={(v) => setRisk('maxMonthlyLoss', v)} min={1000} step={5000} prefix="₹" />
          </FieldRow>
          <FieldRow label="Max Position Size" description="Max % of portfolio in a single position">
            <NumberInput value={localSettings.riskLimits.maxPositionSize} onChange={(v) => setRisk('maxPositionSize', v)} min={1} max={50} step={0.5} suffix="%" />
          </FieldRow>
        </Section>

        {/* Advanced (collapsible) */}
        <div className="bg-surface border border-border rounded-lg overflow-hidden">
          <button
            className="w-full flex items-center justify-between px-5 py-4 text-sm font-semibold text-text-primary hover:bg-surface-2 transition-colors"
            onClick={() => setShowAdvanced((v) => !v)}
          >
            <span>Advanced Settings</span>
            {showAdvanced ? <ChevronUp className="w-4 h-4 text-muted" /> : <ChevronDown className="w-4 h-4 text-muted" />}
          </button>

          {showAdvanced && (
            <div className="px-5 pb-5 space-y-4 border-t border-border pt-4">
              <FieldRow label="Max Sector Concentration">
                <NumberInput value={localSettings.riskLimits.maxSectorConcentration} onChange={(v) => setRisk('maxSectorConcentration', v)} min={5} max={60} step={5} suffix="%" />
              </FieldRow>
              <FieldRow label="Max Correlation" description="Max pairwise correlation between positions">
                <NumberInput value={localSettings.riskLimits.maxCorrelation} onChange={(v) => setRisk('maxCorrelation', v)} min={0} max={1} step={0.05} />
              </FieldRow>
              <FieldRow label="Intraday Max Positions">
                <NumberInput value={localSettings.riskLimits.intradayMaxPositions} onChange={(v) => setRisk('intradayMaxPositions', v)} min={1} max={20} />
              </FieldRow>
              <FieldRow label="Swing Max Positions">
                <NumberInput value={localSettings.riskLimits.swingMaxPositions} onChange={(v) => setRisk('swingMaxPositions', v)} min={1} max={30} />
              </FieldRow>
              <FieldRow label="Long-Term Max Positions">
                <NumberInput value={localSettings.riskLimits.longtermMaxPositions} onChange={(v) => setRisk('longtermMaxPositions', v)} min={1} max={50} />
              </FieldRow>
              <FieldRow label="Rebalancing Threshold" description="Trigger rebalancing when allocation drifts by this %">
                <NumberInput value={localSettings.riskLimits.rebalancingThreshold} onChange={(v) => setRisk('rebalancingThreshold', v)} min={1} max={20} step={0.5} suffix="%" />
              </FieldRow>
            </div>
          )}
        </div>

        {/* Save Button */}
        <div className="flex justify-end pt-2 pb-6">
          <button
            onClick={() => setShowSaveModal(true)}
            className="flex items-center gap-2 px-6 py-2.5 bg-accent text-white rounded-lg font-semibold text-sm hover:bg-accent-hover transition-colors"
          >
            <Save className="w-4 h-4" />
            Save Settings
          </button>
        </div>
      </div>

      {/* Save Confirmation Modal */}
      {showSaveModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-surface border border-border rounded-xl p-6 w-full max-w-sm mx-4 animate-fade-in">
            <div className="flex items-center gap-3 mb-4">
              <Settings className="w-5 h-5 text-accent" />
              <h3 className="font-semibold text-text-primary">Save Settings</h3>
            </div>
            <p className="text-sm text-text-secondary mb-4">
              Changes will take effect immediately.
              {localSettings.tradingMode === 'LIVE' && (
                <span className="text-loss"> You are in LIVE mode — these changes affect real trading.</span>
              )}
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => setShowSaveModal(false)}
                className="flex-1 px-4 py-2 border border-border rounded text-sm text-text-secondary hover:border-muted transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => save.mutate(localSettings)}
                disabled={save.isPending}
                className="flex-1 px-4 py-2 bg-accent text-white rounded text-sm font-semibold hover:bg-accent-hover disabled:opacity-50 transition-colors"
              >
                {save.isPending ? 'Saving...' : 'Confirm Save'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
