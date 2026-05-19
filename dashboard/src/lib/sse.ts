"use client";

import { API_BASE } from "./api";

export type StreamEvent = {
  name: string;
  data: any;
};

/**
 * Stream attribution events from the FastAPI backend. We have to use fetch()
 * because EventSource does not support POST. Returns an `abort` callback.
 */
export function streamAttribution(args: {
  output_id?: string;
  upload?: File;
  totalPayout?: number;
  onEvent: (e: StreamEvent) => void;
  onError?: (err: any) => void;
  onDone?: () => void;
}) {
  const ctrl = new AbortController();
  const url =
    args.output_id !== undefined
      ? `${API_BASE}/api/attribute/${encodeURIComponent(args.output_id)}?total_payout=${args.totalPayout ?? 1.0}`
      : `${API_BASE}/api/attribute/upload?total_payout=${args.totalPayout ?? 1.0}`;

  const headers: HeadersInit = {};
  let body: BodyInit | undefined;
  if (args.upload) {
    const fd = new FormData();
    fd.append("file", args.upload);
    body = fd;
  }

  (async () => {
    try {
      const r = await fetch(url, {
        method: "POST",
        headers,
        body,
        signal: ctrl.signal,
      });
      if (!r.ok || !r.body) {
        throw new Error(`stream → ${r.status}`);
      }
      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        // Parse SSE messages separated by \n\n
        let idx;
        while ((idx = buf.indexOf("\n\n")) !== -1) {
          const block = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          let event = "message";
          let data = "";
          for (const line of block.split("\n")) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            else if (line.startsWith("data:")) data += line.slice(5).trim();
          }
          if (data) {
            try {
              args.onEvent({ name: event, data: JSON.parse(data) });
            } catch {
              args.onEvent({ name: event, data });
            }
          }
        }
      }
      args.onDone?.();
    } catch (err) {
      if ((err as any)?.name !== "AbortError") {
        args.onError?.(err);
      }
    }
  })();

  return () => ctrl.abort();
}
