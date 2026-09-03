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
  callbackRef.current = callback;

  const handleMessage: ChannelCallback<T> = useCallback((data: T) => {
    setLastMessage(data);
    callbackRef.current?.(data);
  }, []);

  useEffect(() => {
    // Connect if not already connected
    try {
      wsClient.connect();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'WebSocket connection failed');
    }

    // Subscribe to connection state changes
    const unlistenConnection = wsClient.onConnectionChange((connected) => {
      setIsConnected(connected);
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
