// ─── Generic API Response ────────────────────────────────────────────────────
export interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
  timestamp: string;
}

// ─── Portfolio ────────────────────────────────────────────────────────────────
export interface PortfolioOverview {
  totalCapital: number;
  investedCapital: number;
  cashBalance: number;
  currentValue: number;
  todayPnl: number;
  todayPnlPct: number;
  weeklyPnl: number;
  weeklyPnlPct: number;
  monthlyPnl: number;
  monthlyPnlPct: number;
  totalReturn: number;
  totalReturnPct: number;
  maxDrawdown: number;
  sharpeRatio: number;
  sortinoRatio: number;
  portfolioVol: number;
  beta: number;
  lastUpdated: string;
}

export interface Position {
  id: string;
  symbol: string;
  exchange: string;
  qty: number;
  avgPrice: number;
  cmp: number;
  marketValue: number;
  unrealizedPnl: number;
  unrealizedPnlPct: number;
  strategy: 'longterm' | 'swing' | 'intraday';
  sector: string;
  portfolioWeight: number;
  stopLoss: number;
  target: number;
  entryDate: string;
  daysHeld: number;
}

export interface Order {
  id: string;
  symbol: string;
  direction: 'BUY' | 'SELL';
  qty: number;
  orderType: 'MARKET' | 'LIMIT' | 'SL' | 'SL-M';
  price?: number;
  triggerPrice?: number;
  status: 'PENDING' | 'OPEN' | 'COMPLETE' | 'CANCELLED' | 'REJECTED';
  filledQty: number;
  avgFillPrice: number;
  strategy: string;
  placedAt: string;
  updatedAt: string;
  rejectionReason?: string;
}

export interface Trade {
  id: string;
  symbol: string;
  strategy: 'longterm' | 'swing' | 'intraday';
  direction: 'LONG' | 'SHORT';
  qty: number;
  entryPrice: number;
  exitPrice: number;
  entryDate: string;
  exitDate: string;
  holdPeriodDays: number;
  grossPnl: number;
  netPnl: number;
  costs: number;
  slippage: number;
  exitReason: string;
  outcome: 'WIN' | 'LOSS' | 'BREAKEVEN';
  auditTrail?: AuditEntry[];
}

export interface AuditEntry {
  timestamp: string;
  action: string;
  details: string;
  orderId?: string;
}

export interface Signal {
  id: string;
  symbol: string;
  strategy: string;
  direction: 'LONG' | 'SHORT';
  score: number;
  strength: 'STRONG' | 'MODERATE' | 'WEAK';
  entryPrice: number;
  stopLoss: number;
  target: number;
  expectedReturn: number;
  expectedHoldDays: number;
  generatedAt: string;
  status: 'PENDING' | 'ENTERED' | 'EXPIRED' | 'REJECTED';
}

// ─── Capital Allocation ───────────────────────────────────────────────────────
export interface CapitalAllocation {
  totalCapital: number;
  longTermCapital: number;
  longTermPct: number;
  swingCapital: number;
  swingPct: number;
  intradayCapital: number;
  intradayPct: number;
  cashBuffer: number;
  cashBufferPct: number;
  longTermRiskPct: number;
  swingRiskPct: number;
  intradayRiskPct: number;
}

export interface AllocationResult {
  id: string;
  contribution: number;
  recommendedAllocation: CapitalAllocation;
  explanation: string;
  rationale: string[];
  calculatedAt: string;
  status: 'PENDING' | 'EXECUTED' | 'EXPIRED';
}

// ─── Strategy ─────────────────────────────────────────────────────────────────
export type StrategyName = 'intraday' | 'swing' | 'longterm';
export type StrategyStatus = 'HEALTHY' | 'REDUCED' | 'PAUSED' | 'DISABLED';

export interface StrategyPerformance {
  strategy: StrategyName;
  totalReturn: number;
  return30d: number;
  return3m: number;
  return1y: number;
  sharpe: number;
  sortino: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  profitFactor: number;
  maxDrawdown: number;
  totalTrades: number;
  activeTrades: number;
}

export interface StrategyHealth {
  strategy: StrategyName;
  status: StrategyStatus;
  lastSignalAt: string;
  activePositions: number;
  positionLimit: number;
  capitalAllocated: number;
  capitalLimit: number;
  recentReturn30d: number;
  drawdown: number;
  isManualOverride: boolean;
  overrideReason?: string;
}

// ─── Market ───────────────────────────────────────────────────────────────────
export type RegimeType = 'BULL_TRENDING' | 'BEAR_TRENDING' | 'BULL_VOLATILE' | 'BEAR_VOLATILE' | 'SIDEWAYS_LOW_VOL' | 'SIDEWAYS_HIGH_VOL';

export interface MarketRegime {
  regime: RegimeType;
  confidence: number;
  vix: number;
  trend: 'UP' | 'DOWN' | 'FLAT';
  breadth: number;
  updatedAt: string;
}

export interface IndexData {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePct: number;
  high: number;
  low: number;
  volume: number;
}

export interface SectorData {
  sector: string;
  return1d: number;
  return1w: number;
  return1m: number;
  topGainer: string;
  topLoser: string;
  breadth: number;
}

export interface MarketOverview {
  indices: IndexData[];
  regime: MarketRegime;
  sectors: SectorData[];
  breadthAbove50dma: number;
  breadthAbove200dma: number;
  advanceDecline: { advances: number; declines: number; unchanged: number };
  topGainers: { symbol: string; changePct: number }[];
  topLosers: { symbol: string; changePct: number }[];
  marketStatus: 'OPEN' | 'CLOSED' | 'PRE_OPEN';
  nextOpen?: string;
}

// ─── Risk ─────────────────────────────────────────────────────────────────────
export interface RiskState {
  portfolioDrawdown: number;
  maxAllowedDrawdown: number;
  dailyLoss: number;
  maxDailyLoss: number;
  weeklyLoss: number;
  maxWeeklyLoss: number;
  monthlyLoss: number;
  maxMonthlyLoss: number;
  activeBreach: RiskBreach[];
  killSwitchActive: boolean;
  var95: number;
  cvar95: number;
  concentrationRisk: number;
}

export interface RiskBreach {
  type: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  message: string;
  triggeredAt: string;
  resolvedAt?: string;
}

export interface RiskLimits {
  maxPortfolioDrawdown: number;
  maxDailyLoss: number;
  maxWeeklyLoss: number;
  maxMonthlyLoss: number;
  maxPositionSize: number;
  maxSectorConcentration: number;
  maxCorrelation: number;
  intradayMaxPositions: number;
  swingMaxPositions: number;
  longtermMaxPositions: number;
  rebalancingThreshold: number;
}

export interface RiskContribution {
  symbol: string;
  strategy: string;
  riskContributionPct: number;
  marketValue: number;
  beta: number;
}

// ─── Models ───────────────────────────────────────────────────────────────────
export type ModelStatus = 'ACTIVE' | 'SHADOW' | 'RETIRED' | 'TRAINING';

export interface ModelVersion {
  id: string;
  name: string;
  strategy: StrategyName;
  version: string;
  trainStart: string;
  trainEnd: string;
  oosStart: string;
  oosEnd: string;
  oosSharpe: number;
  oosReturn: number;
  oosMaxDrawdown: number;
  calibrationScore: number;
  status: ModelStatus;
  deployedAt?: string;
  features: string[];
}

export interface ModelHealth {
  modelId: string;
  driftDetected: boolean;
  driftScore: number;
  lastCalibrationAt: string;
  nextCalibrationDue: string;
  featureImportance: { feature: string; importance: number }[];
  performanceDegradation: number;
}

// ─── System Health ────────────────────────────────────────────────────────────
export type ComponentStatusType = 'HEALTHY' | 'DEGRADED' | 'DOWN' | 'UNKNOWN';

export interface ComponentStatus {
  name: string;
  status: ComponentStatusType;
  lastPing: string;
  latencyMs: number;
  lastError?: string;
  uptime: number;
}

export interface SystemHealth {
  overall: ComponentStatusType;
  components: {
    api: ComponentStatus;
    database: ComponentStatus;
    redis: ComponentStatus;
    broker: ComponentStatus;
    websocket: ComponentStatus;
    marketData: ComponentStatus;
    tradingEngine: ComponentStatus;
  };
  killSwitchActive: boolean;
  tradingMode: 'PAPER' | 'LIVE';
  lastReconciliationAt: string;
  reconciliationResult: 'CLEAN' | 'MISMATCH' | 'PENDING';
}

// ─── User Settings ────────────────────────────────────────────────────────────
export interface UserSettings {
  tradingMode: 'PAPER' | 'LIVE';
  monthlyCapital: number;
  riskTolerance: 'LOW' | 'MEDIUM' | 'HIGH';
  strategies: {
    intraday: boolean;
    swing: boolean;
    longterm: boolean;
  };
  autoExecution: boolean;
  riskLimits: RiskLimits;
  sectorLimits: Record<string, number>;
  rebalancingThreshold: number;
  notifications: {
    riskAlerts: boolean;
    tradeExecutions: boolean;
    dailySummary: boolean;
    weeklyReport: boolean;
  };
}

// ─── Analytics & Charts ───────────────────────────────────────────────────────
export interface EquityCurvePoint {
  date: string;
  portfolio: number;
  nifty: number;
  buyHold?: number;
}

export interface DrawdownPoint {
  date: string;
  drawdown: number;
  portfolioValue: number;
}

export interface MonthlyReturn {
  year: number;
  month: number;
  return: number;
  strategy?: string;
}

// ─── Research / Backtesting ───────────────────────────────────────────────────
export interface BacktestConfig {
  strategy: StrategyName;
  startDate: string;
  endDate: string;
  initialCapital: number;
  costModel: 'ZERO' | 'REALISTIC' | 'CONSERVATIVE';
  parameters?: Record<string, unknown>;
}

export interface BacktestMetrics {
  totalReturn: number;
  annualizedReturn: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  maxDrawdown: number;
  maxDrawdownDuration: number;
  winRate: number;
  profitFactor: number;
  avgWin: number;
  avgLoss: number;
  totalTrades: number;
  annualizedVol: number;
  beta: number;
  alpha: number;
  informationRatio: number;
}

export interface BacktestResult {
  id: string;
  config: BacktestConfig;
  metrics: BacktestMetrics;
  equityCurve: EquityCurvePoint[];
  drawdowns: DrawdownPoint[];
  monthlyReturns: MonthlyReturn[];
  trades: Trade[];
  status: 'PENDING' | 'RUNNING' | 'COMPLETE' | 'FAILED';
  progress?: number;
  startedAt: string;
  completedAt?: string;
  errorMessage?: string;
}

export interface WalkForwardResult {
  id: string;
  strategy: StrategyName;
  windows: WalkForwardWindow[];
  aggregateMetrics: BacktestMetrics;
  consistency: number;
  robustnessScore: number;
}

export interface WalkForwardWindow {
  trainStart: string;
  trainEnd: string;
  testStart: string;
  testEnd: string;
  metrics: BacktestMetrics;
}

export interface MonteCarloResult {
  id: string;
  backtestId: string;
  simulations: number;
  percentiles: {
    p5: number;
    p25: number;
    p50: number;
    p75: number;
    p95: number;
  };
  maxDrawdownDistribution: { drawdown: number; probability: number }[];
  finalReturnDistribution: { return: number; probability: number }[];
  ruinProbability: number;
  expectedReturn: number;
}

// ─── Notifications ────────────────────────────────────────────────────────────
export interface Notification {
  id: string;
  type: 'RISK_ALERT' | 'TRADE_EXEC' | 'SYSTEM' | 'STRATEGY' | 'INFO';
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  title: string;
  message: string;
  read: boolean;
  createdAt: string;
}

// ─── Audit ────────────────────────────────────────────────────────────────────
export interface AuditLog {
  id: string;
  timestamp: string;
  action: string;
  entityType: string;
  entityId: string;
  user: string;
  ip?: string;
  before?: Record<string, unknown>;
  after?: Record<string, unknown>;
  details?: string;
}

// ─── WebSocket Messages ───────────────────────────────────────────────────────
export interface WsMessage<T = unknown> {
  channel: string;
  type: string;
  data: T;
  timestamp: string;
}

export interface PortfolioUpdateMessage {
  currentValue: number;
  todayPnl: number;
  todayPnlPct: number;
  positions: Pick<Position, 'id' | 'symbol' | 'cmp' | 'unrealizedPnl' | 'unrealizedPnlPct'>[];
}

export interface RiskAlertMessage {
  breach: RiskBreach;
  riskState: Partial<RiskState>;
}
