import type { WsMessage, PortfolioUpdateMessage, RiskAlertMessage } from '@/types';

export type ChannelName =
  | 'portfolio_update'
  | 'order_update'
  | 'risk_alert'
  | 'strategy_update'
  | 'market_data';

export type ChannelCallback<T = unknown> = (data: T) => void;

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws';
const HEARTBEAT_INTERVAL_MS = 30_000;
const MAX_RECONNECT_ATTEMPTS = 5;
const BASE_RECONNECT_DELAY_MS = 1_000;

class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string = WS_URL;
  private subscribers: Map<string, Set<ChannelCallback>> = new Map();
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private isIntentionallyClosed = false;
  private connectionListeners: Set<(connected: boolean) => void> = new Set();

  // ─── Connection ─────────────────────────────────────────────────────────────
  connect(url?: string): void {
    if (url) this.url = url;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return;

    this.isIntentionallyClosed = false;
    this._createConnection();
  }

  private _createConnection(): void {
    try {
      this.ws = new WebSocket(this.url);
      this.ws.onopen = this._onOpen.bind(this);
      this.ws.onclose = this._onClose.bind(this);
      this.ws.onerror = this._onError.bind(this);
      this.ws.onmessage = this._onMessage.bind(this);
    } catch (err) {
      console.error('[WS] Failed to create connection:', err);
      this._scheduleReconnect();
    }
  }

  private _onOpen(): void {
    console.log('[WS] Connected');
    this.reconnectAttempts = 0;
    this._notifyConnectionListeners(true);
    this._startHeartbeat();

    // Re-subscribe all active channels
    for (const channel of this.subscribers.keys()) {
      this._sendSubscribe(channel);
    }
  }

  private _onClose(event: CloseEvent): void {
    console.log(`[WS] Closed: code=${event.code}`);
    this._stopHeartbeat();
    this._notifyConnectionListeners(false);

    if (!this.isIntentionallyClosed) {
      this._scheduleReconnect();
    }
  }

  private _onError(event: Event): void {
    console.error('[WS] Error:', event);
  }

  private _onMessage(event: MessageEvent): void {
    try {
      const msg: WsMessage = JSON.parse(event.data as string);

      if (msg.type === 'pong') return; // heartbeat ack

      const subs = this.subscribers.get(msg.channel);
      if (subs) {
        for (const cb of subs) {
          cb(msg.data);
        }
      }
    } catch (err) {
      console.error('[WS] Failed to parse message:', err);
    }
  }

  // ─── Reconnect ──────────────────────────────────────────────────────────────
  private _scheduleReconnect(): void {
    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      console.warn('[WS] Max reconnect attempts reached');
      return;
    }

    const delay = BASE_RECONNECT_DELAY_MS * Math.pow(2, this.reconnectAttempts);
    this.reconnectAttempts++;
    console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    this.reconnectTimer = setTimeout(() => {
      this._createConnection();
    }, delay);
  }

  // ─── Heartbeat ──────────────────────────────────────────────────────────────
  private _startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping', timestamp: new Date().toISOString() }));
      }
    }, HEARTBEAT_INTERVAL_MS);
  }

  private _stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  // ─── Subscribe / Unsubscribe ─────────────────────────────────────────────────
  subscribe<T>(channel: ChannelName, callback: ChannelCallback<T>): () => void {
    if (!this.subscribers.has(channel)) {
      this.subscribers.set(channel, new Set());
      // Send subscribe message if already connected
      if (this.ws?.readyState === WebSocket.OPEN) {
        this._sendSubscribe(channel);
      }
    }
    const cb = callback as ChannelCallback;
    this.subscribers.get(channel)!.add(cb);

    // Return unsubscribe function
    return () => this.unsubscribe(channel, callback);
  }

  unsubscribe<T>(channel: ChannelName, callback: ChannelCallback<T>): void {
    const subs = this.subscribers.get(channel);
    if (!subs) return;

    const cb = callback as ChannelCallback;
    subs.delete(cb);

    if (subs.size === 0) {
      this.subscribers.delete(channel);
      if (this.ws?.readyState === WebSocket.OPEN) {
        this._sendUnsubscribe(channel);
      }
    }
  }

  private _sendSubscribe(channel: string): void {
    this.ws?.send(JSON.stringify({ type: 'subscribe', channel }));
  }

  private _sendUnsubscribe(channel: string): void {
    this.ws?.send(JSON.stringify({ type: 'unsubscribe', channel }));
  }

  // ─── Connection State Listeners ──────────────────────────────────────────────
  onConnectionChange(listener: (connected: boolean) => void): () => void {
    this.connectionListeners.add(listener);
    return () => this.connectionListeners.delete(listener);
  }

  private _notifyConnectionListeners(connected: boolean): void {
    for (const listener of this.connectionListeners) {
      listener(connected);
    }
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  // ─── Typed Channel Helpers ───────────────────────────────────────────────────
  onPortfolioUpdate(callback: ChannelCallback<PortfolioUpdateMessage>): () => void {
    return this.subscribe('portfolio_update', callback);
  }

  onOrderUpdate(callback: ChannelCallback<unknown>): () => void {
    return this.subscribe('order_update', callback);
  }

  onRiskAlert(callback: ChannelCallback<RiskAlertMessage>): () => void {
    return this.subscribe('risk_alert', callback);
  }

  onStrategyUpdate(callback: ChannelCallback<unknown>): () => void {
    return this.subscribe('strategy_update', callback);
  }

  onMarketData(callback: ChannelCallback<unknown>): () => void {
    return this.subscribe('market_data', callback);
  }

  // ─── Disconnect ──────────────────────────────────────────────────────────────
  disconnect(): void {
    this.isIntentionallyClosed = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this._stopHeartbeat();
    this.ws?.close(1000, 'Client disconnect');
    this.ws = null;
  }
}

// Singleton instance
const wsClient = new WebSocketClient();
export default wsClient;
