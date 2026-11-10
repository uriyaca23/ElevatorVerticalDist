"use client";

import { useEffect, useRef, useState } from "react";

interface State {
  url: string | null;
  uploadUrl: string | null;
  error?: string;
}

export function GramushkaViewer({ sessionId }: { sessionId: number }) {
  const [state, setState] = useState<State | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const refresh = () =>
    fetch(`/api/sessions/${sessionId}/gramushka`, { cache: "no-store" })
      .then((r) => r.json() as Promise<State>)
      .then(setState);

  useEffect(() => {
    let alive = true;
    fetch(`/api/sessions/${sessionId}/gramushka`, { cache: "no-store" })
      .then((r) => r.json() as Promise<State>)
      .then((s) => alive && setState(s));
    return () => {
      alive = false;
    };
  }, [sessionId]);

  function upload(file: File) {
    if (!state?.uploadUrl) return;
    setUploadError(null);
    setProgress(0);
    // XHR rather than fetch for upload progress; plans are tens of MB.
    const xhr = new XMLHttpRequest();
    xhr.open("POST", state.uploadUrl);
    xhr.setRequestHeader("Content-Type", "application/pdf");
    xhr.upload.onprogress = (e) => e.lengthComputable && setProgress(e.loaded / e.total);
    xhr.onload = () => {
      setProgress(null);
      if (xhr.status >= 200 && xhr.status < 300) refresh();
      else setUploadError(`Upload failed (${xhr.status})`);
    };
    xhr.onerror = () => {
      setProgress(null);
      setUploadError("Upload failed (network error)");
    };
    xhr.send(file);
  }

  return (
    <details open={!!state?.url} className="group rounded-lg border border-neutral-200 bg-white p-4">
      <summary className="flex cursor-pointer list-none items-center gap-3">
        <span className="text-neutral-400 transition group-open:rotate-90">▶</span>
        <h2 className="font-semibold">Garamushka</h2>
        {state && !state.error && <span className="text-sm text-neutral-500">{state.url ? "PDF attached" : "none yet"}</span>}
        <input
          ref={input}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) upload(f);
            e.target.value = "";
          }}
        />
        {state && !state.error && (
          <button
            onClick={(e) => {
              e.preventDefault();
              input.current?.click();
            }}
            disabled={progress != null}
            className="rounded-md border border-neutral-300 px-2.5 py-1 text-sm hover:bg-neutral-50 disabled:opacity-50"
          >
            {state.url ? "Replace PDF" : "Upload PDF"}
          </button>
        )}
        {progress != null && <span className="text-sm text-neutral-500">Uploading… {Math.round(progress * 100)}%</span>}
        {uploadError && <span className="text-sm text-red-600">{uploadError}</span>}
      </summary>
      <div className="mt-3" />
      {!state && <p className="text-sm text-neutral-500">Checking…</p>}
      {state?.error && <p className="text-sm text-red-600">{state.error}</p>}
      {state && !state.error && !state.url && (
        <p className="text-sm text-neutral-500">No garamushka uploaded for this experiment yet.</p>
      )}
      {state?.url && <iframe src={state.url} title="Garamushka" className="h-[70vh] w-full rounded border border-neutral-200" />}
    </details>
  );
}
