'use client';

import { useState } from 'react';
import { Header } from '@/components/layout/Header';
import { StatusBadge } from '@/components/common/StatusBadge';
import { useQuery } from '@tanstack/react-query';
import { researchApi } from '@/lib/api';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
} from 'recharts';
import { clsx } from 'clsx';
import { Brain, AlertTriangle, CheckCircle } from 'lucide-react';
import type { ModelVersion, ModelHealth } from '@/types';

const STRATEGY_LABELS: Record<string, string> = {
  intraday: 'Intraday',
  swing: 'Swing',
  longterm: 'Long-Term',
};

export default function ModelsPage() {
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  const { data: models, isLoading } = useQuery({
    queryKey: ['models'],
    queryFn: () => researchApi.getModels(),
    staleTime: 120_000,
  });

  const { data: modelHealth } = useQuery({
    queryKey: ['model-health', selectedModelId],
    queryFn: () => researchApi.getModelHealth(selectedModelId!),
    enabled: !!selectedModelId,
    staleTime: 60_000,
  });

  const selectedModel = models?.find((m) => m.id === selectedModelId);

  return (
    <div className="flex flex-col h-full">
      <Header title="Models" />

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Active Models Table */}
        <section className="bg-surface border border-border rounded-lg">
          <div className="flex items-center gap-2 px-4 py-3 border-b border-border">
            <Brain className="w-4 h-4 text-accent" />
            <h3 className="text-sm font-semibold text-text-primary">ML Models</h3>
          </div>

          {isLoading ? (
            <div className="p-8 text-center text-muted animate-pulse">Loading models...</div>
          ) : !models || models.length === 0 ? (
            <div className="py-10 text-center text-muted text-sm">No models registered</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Strategy</th>
                    <th>Version</th>
                    <th>Train Period</th>
                    <th>OOS Sharpe</th>
                    <th>OOS Return</th>
                    <th>OOS Max DD</th>
                    <th>Calibration</th>
                    <th>Status</th>
                    <th>Deployed</th>
                    <th>Details</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((model: ModelVersion) => (
                    <tr
                      key={model.id}
                      className={clsx(selectedModelId === model.id && 'bg-surface-2')}
                    >
                      <td className="font-semibold text-text-primary">{model.name}</td>
                      <td className="text-xs text-text-secondary">{STRATEGY_LABELS[model.strategy] ?? model.strategy}</td>
                      <td className="font-mono text-xs text-muted">{model.version}</td>
                      <td className="text-xs text-muted">
                        {new Date(model.trainStart).toLocaleDateString('en-IN', { month: 'short', year: '2-digit' })}
                        {' – '}
                        {new Date(model.trainEnd).toLocaleDateString('en-IN', { month: 'short', year: '2-digit' })}
                      </td>
                      <td className={clsx('font-mono tabular-nums', model.oosSharpe >= 1 ? 'text-profit' : model.oosSharpe >= 0 ? 'text-text-primary' : 'text-loss')}>
                        {model.oosSharpe.toFixed(2)}
                      </td>
                      <td className={clsx('font-mono tabular-nums', model.oosReturn >= 0 ? 'text-profit' : 'text-loss')}>
                        {model.oosReturn >= 0 ? '+' : ''}{model.oosReturn.toFixed(2)}%
                      </td>
                      <td className="font-mono tabular-nums text-loss">{model.oosMaxDrawdown.toFixed(2)}%</td>
                      <td>
                        <div className="flex items-center gap-1.5">
                          <div className="w-12 h-1.5 bg-border rounded-full overflow-hidden">
                            <div
                              className={clsx('h-full', model.calibrationScore >= 0.7 ? 'bg-profit' : model.calibrationScore >= 0.4 ? 'bg-warning' : 'bg-loss')}
                              style={{ width: `${model.calibrationScore * 100}%` }}
                            />
                          </div>
                          <span className="text-xs font-mono text-text-secondary">{(model.calibrationScore * 100).toFixed(0)}</span>
                        </div>
                      </td>
                      <td><StatusBadge status={model.status} size="sm" /></td>
                      <td className="text-xs text-muted">
                        {model.deployedAt ? new Date(model.deployedAt).toLocaleDateString('en-IN') : '—'}
                      </td>
                      <td>
                        <button
                          className="text-xs text-accent hover:text-accent-hover"
                          onClick={() => setSelectedModelId(selectedModelId === model.id ? null : model.id)}
                        >
                          {selectedModelId === model.id ? 'Close' : 'Inspect'}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* Model Detail Panel */}
        {selectedModel && (
          <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Feature Importance */}
            <div className="bg-surface border border-border rounded-lg p-4">
              <h3 className="text-sm font-semibold text-text-primary mb-4">
                Feature Importance — {selectedModel.name}
              </h3>
              {modelHealth?.featureImportance && modelHealth.featureImportance.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart
                    data={modelHealth.featureImportance.slice(0, 10).sort((a, b) => b.importance - a.importance)}
                    layout="vertical"
                    margin={{ top: 0, right: 8, left: 100, bottom: 0 }}
                  >
                    <XAxis type="number" tick={{ fill: '#6b7280', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v: number) => v.toFixed(2)} />
                    <YAxis type="category" dataKey="feature" tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} tickLine={false} width={95} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#111118', border: '1px solid #1f2028', borderRadius: 6, fontSize: 12 }}
                      formatter={(v: number) => [v.toFixed(3), 'Importance']}
                    />
                    <Bar dataKey="importance" radius={[0, 3, 3, 0]}>
                      {(modelHealth.featureImportance ?? []).map((_, i) => (
                        <Cell key={i} fill={`hsl(${210 + i * 15}, 70%, 55%)`} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-center text-muted text-sm py-8">No feature importance data</div>
              )}
            </div>

            {/* Drift Detection */}
            <div className="bg-surface border border-border rounded-lg p-4">
              <h3 className="text-sm font-semibold text-text-primary mb-4">Drift Detection</h3>
              {modelHealth ? (
                <div className="space-y-4">
                  <div className={clsx(
                    'flex items-center gap-3 p-3 rounded-lg border',
                    modelHealth.driftDetected
                      ? 'bg-loss/10 border-loss/30 text-loss'
                      : 'bg-profit/10 border-profit/30 text-profit',
                  )}>
                    {modelHealth.driftDetected ? (
                      <AlertTriangle className="w-5 h-5 shrink-0" />
                    ) : (
                      <CheckCircle className="w-5 h-5 shrink-0" />
                    )}
                    <div>
                      <p className="text-sm font-semibold">
                        {modelHealth.driftDetected ? 'Drift Detected' : 'No Drift Detected'}
                      </p>
                      <p className="text-xs opacity-80">Score: {modelHealth.driftScore.toFixed(3)}</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="bg-surface-2 rounded p-2.5">
                      <p className="text-muted mb-1">Last Calibration</p>
                      <p className="font-mono text-text-secondary">
                        {new Date(modelHealth.lastCalibrationAt).toLocaleDateString('en-IN')}
                      </p>
                    </div>
                    <div className="bg-surface-2 rounded p-2.5">
                      <p className="text-muted mb-1">Next Due</p>
                      <p className="font-mono text-text-secondary">
                        {new Date(modelHealth.nextCalibrationDue).toLocaleDateString('en-IN')}
                      </p>
                    </div>
                    <div className="bg-surface-2 rounded p-2.5 col-span-2">
                      <p className="text-muted mb-1">Performance Degradation</p>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 progress-bar">
                          <div
                            className={clsx('progress-bar-fill', modelHealth.performanceDegradation > 0.15 ? 'bg-loss' : 'bg-accent')}
                            style={{ width: `${Math.min(modelHealth.performanceDegradation * 100, 100)}%` }}
                          />
                        </div>
                        <span className={clsx('font-mono', modelHealth.performanceDegradation > 0.15 ? 'text-loss' : 'text-text-secondary')}>
                          {(modelHealth.performanceDegradation * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center text-muted text-sm py-8">Select a model to view drift status</div>
              )}
            </div>
          </section>
        )}

        {/* Model Features List */}
        {selectedModel && selectedModel.features.length > 0 && (
          <section className="bg-surface border border-border rounded-lg p-4">
            <h3 className="text-sm font-semibold text-text-primary mb-3">Features ({selectedModel.features.length})</h3>
            <div className="flex flex-wrap gap-2">
              {selectedModel.features.map((feature) => (
                <span key={feature} className="px-2 py-1 bg-surface-2 border border-border rounded text-xs text-text-secondary font-mono">
                  {feature}
                </span>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
