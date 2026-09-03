import axios, { AxiosInstance, InternalAxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';
import type {
  ApiResponse,
  PortfolioOverview,
  Position,
  CapitalAllocation,
  AllocationResult,
  Trade,
  Signal,
  StrategyHealth,
  StrategyPerformance,
  MarketOverview,
  MarketRegime,
  SectorData,
  RiskState,
  RiskLimits,
  SystemHealth,
  UserSettings,
  BacktestConfig,
  BacktestResult,
  ModelVersion,
  ModelHealth,
  WalkForwardResult,
  AuditLog,
  Notification,
  StrategyName,
} from '@/types';

// ─── Axios Instance ───────────────────────────────────────────────────────────
const api: AxiosInstance = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Auth token injection
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem('algodollar_token');
      if (token && config.headers) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }
    return config;
  },
  (error: AxiosError) => Promise.reject(error),
);

// Response error handling
api.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      if (typeof window !== 'undefined') {
        localStorage.removeItem('algodollar_token');
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  },
);

// Helper to unwrap ApiResponse
async function get<T>(url: string, params?: Record<string, unknown>): Promise<T> {
  const res = await api.get<ApiResponse<T>>(url, { params });
  return res.data.data;
}

async function post<T>(url: string, data?: unknown): Promise<T> {
  const res = await api.post<ApiResponse<T>>(url, data);
  return res.data.data;
}

async function put<T>(url: string, data?: unknown): Promise<T> {
  const res = await api.put<ApiResponse<T>>(url, data);
  return res.data.data;
}

async function patch<T>(url: string, data?: unknown): Promise<T> {
  const res = await api.patch<ApiResponse<T>>(url, data);
  return res.data.data;
}

// ─── Portfolio ────────────────────────────────────────────────────────────────
export const portfolioApi = {
  getOverview: () => get<PortfolioOverview>('/api/v1/portfolio/overview'),
  getPositions: (strategy?: StrategyName) =>
    get<Position[]>('/api/v1/portfolio/positions', strategy ? { strategy } : undefined),
  getAllocation: () => get<CapitalAllocation>('/api/v1/portfolio/allocation'),
  getPerformance: (period: '1D' | '1W' | '1M' | '3M' | '6M' | '1Y' | '3Y' | 'ALL') =>
    get<PortfolioOverview>('/api/v1/portfolio/performance', { period }),
};

// ─── Allocation ───────────────────────────────────────────────────────────────
export const allocationApi = {
  calculate: (contribution: number) =>
    post<AllocationResult>('/api/v1/allocation/calculate', { contribution }),
  execute: (allocationId: string) =>
    post<AllocationResult>(`/api/v1/allocation/${allocationId}/execute`),
  getHistory: () => get<AllocationResult[]>('/api/v1/allocation/history'),
  getExplanation: (id: string) => get<{ explanation: string; rationale: string[] }>(`/api/v1/allocation/${id}/explanation`),
};

// ─── Trades ───────────────────────────────────────────────────────────────────
export interface TradeFilters {
  startDate?: string;
  endDate?: string;
  strategy?: StrategyName;
  symbol?: string;
  outcome?: 'WIN' | 'LOSS' | 'BREAKEVEN';
  page?: number;
  limit?: number;
}

export interface TradeSummary {
  totalTrades: number;
  winRate: number;
  grossPnl: number;
  netPnl: number;
  totalCosts: number;
  avgSlippage: number;
  byStrategy: Record<string, { count: number; pnl: number; winRate: number }>;
}

export const tradesApi = {
  getTrades: (filters?: TradeFilters) =>
    get<{ trades: Trade[]; total: number }>('/api/v1/trades', filters as Record<string, unknown>),
  getTrade: (id: string) => get<Trade>(`/api/v1/trades/${id}`),
  getSummary: (filters?: TradeFilters) =>
    get<TradeSummary>('/api/v1/trades/summary', filters as Record<string, unknown>),
};

// ─── Strategies ───────────────────────────────────────────────────────────────
export const strategiesApi = {
  getAll: () => get<StrategyHealth[]>('/api/v1/strategies'),
  get: (name: StrategyName) => get<StrategyHealth>(`/api/v1/strategies/${name}`),
  updateStatus: (name: StrategyName, status: string, reason?: string) =>
    patch<StrategyHealth>(`/api/v1/strategies/${name}/status`, { status, reason }),
  getSignals: (name: StrategyName) => get<Signal[]>(`/api/v1/strategies/${name}/signals`),
  getPerformance: (name: StrategyName) =>
    get<StrategyPerformance>(`/api/v1/strategies/${name}/performance`),
};

// ─── Markets ──────────────────────────────────────────────────────────────────
export const marketsApi = {
  getOverview: () => get<MarketOverview>('/api/v1/markets/overview'),
  getRegime: () => get<MarketRegime>('/api/v1/markets/regime'),
  getSectors: () => get<SectorData[]>('/api/v1/markets/sectors'),
  getOpportunities: () => get<Signal[]>('/api/v1/markets/opportunities'),
};

// ─── Settings ─────────────────────────────────────────────────────────────────
export interface CostModel {
  brokerage: number;
  stt: number;
  exchangeCharges: number;
  sebiCharges: number;
  stampDuty: number;
  gst: number;
  totalRoundTripBps: number;
}

export const settingsApi = {
  get: () => get<UserSettings>('/api/v1/settings'),
  update: (settings: Partial<UserSettings>) => put<UserSettings>('/api/v1/settings', settings),
  toggleKillSwitch: (active: boolean) =>
    post<{ active: boolean }>('/api/v1/settings/kill-switch', { active }),
  getCostModel: () => get<CostModel>('/api/v1/settings/cost-model'),
};

// ─── Health ───────────────────────────────────────────────────────────────────
export const healthApi = {
  getHealth: () => get<SystemHealth>('/api/v1/health'),
  getDetailedHealth: () => get<SystemHealth>('/api/v1/health/detailed'),
};

// ─── Research ─────────────────────────────────────────────────────────────────
export const researchApi = {
  runBacktest: (config: BacktestConfig) =>
    post<BacktestResult>('/api/v1/research/backtest', config),
  getBacktest: (id: string) => get<BacktestResult>(`/api/v1/research/backtest/${id}`),
  getBacktests: () => get<BacktestResult[]>('/api/v1/research/backtests'),
  getModels: () => get<ModelVersion[]>('/api/v1/research/models'),
  getModelHealth: (modelId: string) => get<ModelHealth>(`/api/v1/research/models/${modelId}/health`),
  runWalkForward: (config: BacktestConfig & { windows: number }) =>
    post<WalkForwardResult>('/api/v1/research/walk-forward', config),
};

// ─── Risk ─────────────────────────────────────────────────────────────────────
export const riskApi = {
  getState: () => get<RiskState>('/api/v1/risk/state'),
  getLimits: () => get<RiskLimits>('/api/v1/risk/limits'),
  squareOffAll: (strategy?: StrategyName) =>
    post<{ ordersPlaced: number }>('/api/v1/risk/square-off', { strategy }),
};

// ─── Notifications ────────────────────────────────────────────────────────────
export const notificationsApi = {
  getAll: () => get<Notification[]>('/api/v1/notifications'),
  markRead: (id: string) => patch<void>(`/api/v1/notifications/${id}/read`),
  markAllRead: () => patch<void>('/api/v1/notifications/read-all'),
};

// ─── Audit ────────────────────────────────────────────────────────────────────
export interface AuditFilters {
  startDate?: string;
  endDate?: string;
  action?: string;
  entityType?: string;
  page?: number;
  limit?: number;
}

export const auditApi = {
  getLogs: (filters?: AuditFilters) =>
    get<{ logs: AuditLog[]; total: number }>('/api/v1/audit', filters as Record<string, unknown>),
};

export default api;
