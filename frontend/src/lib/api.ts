import axios, { AxiosInstance, InternalAxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';
import type {
  EquityCurvePoint,
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
  RiskLimitsResponse,
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

// ─── Payload shape ────────────────────────────────────────────────────────────
//
// TWO MISMATCHES THAT MADE EVERY CALL RETURN `undefined`
//
// 1. These helpers unwrapped `res.data.data`, assuming every response was
//    wrapped in an `{ data: ... }` envelope. FastAPI returns the payload
//    directly, so `.data` on it was always undefined — every hook in the app
//    received undefined and every component fell back to its placeholder. That
//    is why the dashboard looked like a mock: the real values never arrived.
//
// 2. The backend serialises snake_case (`total_capital`); every type and every
//    component here reads camelCase (`totalCapital`). Even with the envelope
//    fixed, each field would still have been undefined.
//
// Both are handled here rather than by renaming fields on either side: the
// Python API keeps Python conventions, the TypeScript keeps TypeScript ones,
// and exactly one place translates.

function toCamel(key: string): string {
  return key.replace(/_([a-z0-9])/g, (_, c: string) => c.toUpperCase());
}

function toSnake(key: string): string {
  return key.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`);
}

/** Deep key conversion. Arrays and primitives pass through unchanged. */
function convertKeys(value: unknown, convert: (k: string) => string): unknown {
  if (Array.isArray(value)) return value.map((v) => convertKeys(v, convert));
  if (value === null || typeof value !== 'object') return value;
  // Date and other class instances must not be rebuilt as plain objects.
  if (Object.getPrototypeOf(value) !== Object.prototype) return value;
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>).map(([k, v]) => [
      convert(k),
      convertKeys(v, convert),
    ]),
  );
}

/**
 * Unwrap the payload and camelise it.
 *
 * The envelope is still tolerated: a response that genuinely carries
 * `{ data, success }` is unwrapped, anything else is used as-is. Assuming one
 * shape and silently producing undefined for the other is the bug this
 * replaces.
 */
function unwrap<T>(body: unknown): T {
  const payload =
    body !== null &&
    typeof body === 'object' &&
    'data' in (body as Record<string, unknown>) &&
    'success' in (body as Record<string, unknown>)
      ? (body as Record<string, unknown>).data
      : body;
  return convertKeys(payload, toCamel) as T;
}

/** Query params are sent in the casing the backend declares them in. */
function snakeParams(
  params?: Record<string, unknown>,
): Record<string, unknown> | undefined {
  if (!params) return undefined;
  return Object.fromEntries(
    Object.entries(params)
      .filter(([, v]) => v !== undefined && v !== null)
      .map(([k, v]) => [toSnake(k), v]),
  );
}

async function get<T>(url: string, params?: Record<string, unknown>): Promise<T> {
  const res = await api.get(url, { params: snakeParams(params) });
  return unwrap<T>(res.data);
}

async function post<T>(url: string, data?: unknown): Promise<T> {
  const res = await api.post(url, convertKeys(data, toSnake));
  return unwrap<T>(res.data);
}

async function put<T>(url: string, data?: unknown): Promise<T> {
  const res = await api.put(url, convertKeys(data, toSnake));
  return unwrap<T>(res.data);
}

async function patch<T>(url: string, data?: unknown): Promise<T> {
  const res = await api.patch(url, convertKeys(data, toSnake));
  return unwrap<T>(res.data);
}

// ─── Portfolio ────────────────────────────────────────────────────────────────
export const portfolioApi = {
  getOverview: () => get<PortfolioOverview>('/api/v1/portfolio/overview'),
  getPositions: (strategy?: StrategyName) =>
    get<Position[]>('/api/v1/portfolio/positions', strategy ? { strategy } : undefined),
  getAllocation: () => get<CapitalAllocation>('/api/v1/portfolio/allocation'),
  // Returns the EQUITY CURVE, not an overview: the backend's
  // /portfolio/performance is declared `list[EquityCurvePoint]`. Typed as
  // PortfolioOverview, every consumer saw a shape the endpoint never sends.
  getPerformance: (period: '1D' | '1W' | '1M' | '3M' | '6M' | '1Y' | '3Y' | 'ALL') =>
    get<EquityCurvePoint[]>('/api/v1/portfolio/performance', { period }),
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
  getLimits: () => get<RiskLimitsResponse>('/api/v1/risk/limits'),
  squareOffAll: (strategy?: StrategyName) =>
    post<{ ordersPlaced: number }>('/api/v1/risk/square-off', { strategy }),
};

// ─── Notifications ────────────────────────────────────────────────────────────
export const notificationsApi = {
  getAll: () => get<Notification[]>('/api/v1/notifications'),
  markRead: (id: string) => patch<void>(`/api/v1/notifications/${id}/read`),
  markAllRead: () => patch<void>('/api/v1/notifications/read-all'),
};

// ─── Execution decisions ──────────────────────────────────────────────────────
//
// Every execution attempt, INCLUDING every refusal, with the specific gate that
// caused it. This is what lets the UI say
//
//     RELIANCE BUY x12 rejected — sector exposure limit
//
// rather than showing an empty trade list, which reads identically to a quiet
// market, a dead feed and an engaged kill switch.

export interface ExecutionDecision {
  auditId: string;
  timestamp: string;
  tradingMode: string;
  symbol: string | null;
  side: string | null;
  quantity: number | null;
  strategy: string | null;

  outcome: string;
  submitted: boolean;
  /** "RELIANCE BUY x12 rejected — sector exposure limit" */
  headline: string;
  /** The gate, in plain language. */
  reason: string | null;
  /** The numbers behind it, casing preserved. */
  detail: string | null;
  rawReason: string | null;
  failedChecks: string[];
  failedGates: string[];

  killSwitchActive: boolean | null;
  reconciliationState: string | null;
  eligibilityState: string | null;

  brokerOrderId: string | null;
  fillQuantity: number | null;
  averageFillPrice: number | null;
  intendedNotional: number | null;
  edgeScore: number | null;
  expectedReturn: number | null;
  error: string | null;
}

export interface ExecutionDecisionsResponse {
  entries: ExecutionDecision[];
  total: number;
  submitted: number;
  rejected: number;
  source: string;
  /**
   * Set when the trail itself could not be read. "Nothing was attempted" and
   * "we cannot see what was attempted" must never render the same way.
   */
  unavailableReason: string | null;
}

export interface ExecutionAuditFilters {
  limit?: number;
  rejectedOnly?: boolean;
  symbol?: string;
  strategy?: string;
}

export const executionApi = {
  getDecisions: (filters?: ExecutionAuditFilters) =>
    get<ExecutionDecisionsResponse>('/api/v1/audit', {
      limit: filters?.limit,
      rejected_only: filters?.rejectedOnly,
      symbol: filters?.symbol,
      strategy: filters?.strategy,
    } as Record<string, unknown>),
  getDecision: (auditId: string) =>
    get<ExecutionDecision>(`/api/v1/audit/${auditId}`),
};

// Kept for the existing audit page, which lists database AuditLog rows rather
// than execution decisions. The two are different records and are not merged.
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
