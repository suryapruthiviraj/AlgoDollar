'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { portfolioApi, allocationApi } from '@/lib/api';
import { useWebSocket } from '@/hooks/useWebSocket';
import type { PortfolioOverview, Position, AllocationResult, PortfolioUpdateMessage, StrategyName } from '@/types';

// ─── usePortfolio ─────────────────────────────────────────────────────────────
export function usePortfolio(strategyFilter?: StrategyName) {
  const queryClient = useQueryClient();

  const overview = useQuery<PortfolioOverview>({
    queryKey: ['portfolio', 'overview'],
    queryFn: () => portfolioApi.getOverview(),
    refetchInterval: 30_000,
    staleTime: 10_000,
  });

  const positions = useQuery<Position[]>({
    queryKey: ['portfolio', 'positions', strategyFilter],
    queryFn: () => portfolioApi.getPositions(strategyFilter),
    refetchInterval: 30_000,
    staleTime: 10_000,
  });

  const allocation = useQuery({
    queryKey: ['portfolio', 'allocation'],
    queryFn: () => portfolioApi.getAllocation(),
    refetchInterval: 60_000,
    staleTime: 30_000,
  });

  const submitContribution = useMutation({
    mutationFn: (contribution: number) => allocationApi.calculate(contribution),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolio'] });
    },
  });

  return {
    overview: overview.data,
    positions: positions.data ?? [],
    allocation: allocation.data,
    isLoading: overview.isLoading || positions.isLoading,
    error: overview.error || positions.error,
    submitContribution,
  };
}

// ─── useAllocation ────────────────────────────────────────────────────────────
export function useAllocation() {
  const queryClient = useQueryClient();
  const [pendingAllocation, setPendingAllocation] = useState<AllocationResult | null>(null);

  const calculate = useMutation({
    mutationFn: (contribution: number) => allocationApi.calculate(contribution),
    onSuccess: (data) => {
      setPendingAllocation(data);
    },
  });

  const execute = useMutation({
    mutationFn: (allocationId: string) => allocationApi.execute(allocationId),
    onSuccess: () => {
      setPendingAllocation(null);
      queryClient.invalidateQueries({ queryKey: ['portfolio'] });
    },
  });

  const history = useQuery({
    queryKey: ['allocation', 'history'],
    queryFn: () => allocationApi.getHistory(),
    staleTime: 60_000,
  });

  return {
    calculate,
    execute,
    history: history.data ?? [],
    pendingAllocation,
    setPendingAllocation,
  };
}

// ─── useRealTimePortfolio ─────────────────────────────────────────────────────
export function useRealTimePortfolio() {
  const queryClient = useQueryClient();
  const [liveUpdate, setLiveUpdate] = useState<PortfolioUpdateMessage | null>(null);

  const { isConnected } = useWebSocket<PortfolioUpdateMessage>(
    'portfolio_update',
    (data) => {
      setLiveUpdate(data);
      // Merge live data into the overview query cache
      queryClient.setQueryData<PortfolioOverview>(['portfolio', 'overview'], (old) => {
        if (!old) return old;
        return {
          ...old,
          currentValue: data.currentValue,
          todayPnl: data.todayPnl,
          todayPnlPct: data.todayPnlPct,
        };
      });
    },
  );

  return {
    liveUpdate,
    isConnected,
    currentValue: liveUpdate?.currentValue ?? null,
    todayPnl: liveUpdate?.todayPnl ?? null,
    todayPnlPct: liveUpdate?.todayPnlPct ?? null,
  };
}
