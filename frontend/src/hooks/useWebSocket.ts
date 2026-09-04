'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import wsClient, { type ChannelName, type ChannelCallback } from '@/lib/websocket';

interface UseWebSocketReturn<T> {
  isConnected: boolean;
  lastMessage: T | null;
  error: string | null;
}

export function useWebSocket<T = unknown>(
  channel: ChannelName,
  callback?: (data: T) => void,
): UseWebSocketReturn<T> {
  const [isConnected, setIsConnected] = useState(wsClient.isConnected);
  const [lastMessage, setLastMessage] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const callbackRef = useRef(callback);
  // Written in an effect, not during render. Mutating a ref while rendering is
  // unsafe under concurrent React: a render can be discarded and re-run, so the
  // ref could be left holding a callback from an abandoned render.
  useEffect(() => {
    callbackRef.current = callback;
  });

  const handleMessage: ChannelCallback<T> = useCallback((data: T) => {
    setLastMessage(data);
    callbackRef.current?.(data);
  }, []);

  useEffect(() => {
    // Connect if not already connected.
    //
    // This was wrapped in try/catch with `setError` in the handler, but
    // `wsClient.connect()` CANNOT THROW: `_createConnection()` catches its own
    // failures, logs them and schedules a reconnect. So the catch was dead code
    // and `error` was null no matter what the feed was actually doing — the
    // hook advertised error reporting it did not perform.
    wsClient.connect();

    // Subscribe to connection state changes. `error` is now driven from here,
    // which is the only place that learns the real socket state — and being an
    // async callback rather than the effect body, it also does not force the
    // cascading re-render the old synchronous setError caused.
    const unlistenConnection = wsClient.onConnectionChange((connected) => {
      setIsConnected(connected);
      setError(connected ? null : 'WebSocket disconnected — live data may be stale');
    });

    // Subscribe to channel
    const unsubscribe = wsClient.subscribe<T>(channel, handleMessage);

    return () => {
      unsubscribe();
      unlistenConnection();
    };
  }, [channel, handleMessage]);

  return { isConnected, lastMessage, error };
}

export default useWebSocket;
