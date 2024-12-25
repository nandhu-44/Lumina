import { useCallback, useEffect } from "react";

export function useWebSocket(url, onMessage) {
    const connect = useCallback(() => {
        const ws = new WebSocket(url);

        ws.onopen = () => {
            console.log("WebSocket connected");
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                onMessage(event);
            } catch (error) {
                console.error("Failed to parse WebSocket message:", error);
            }
        };

        ws.onerror = (error) => {
            console.error("WebSocket error:", error);
        };

        ws.onclose = () => {
            console.log("WebSocket disconnected, attempting to reconnect...");
            setTimeout(connect, 1000);
        };

        return ws;
    }, [url, onMessage]);

    useEffect(() => {
        const ws = connect();
        return () => {
            ws.close();
        };
    }, [connect]);
}
