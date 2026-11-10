"use client";

import { useEffect, useRef } from "react";

interface Props {
  open: boolean;
  title: string;
  children: React.ReactNode;
  confirmLabel: string;
  tone?: "danger" | "primary";
  busy?: boolean;
  error?: string | null;
  onConfirm: () => void;
  onCancel: () => void;
}

/** Modal confirmation popup. Esc or the backdrop cancels; Cancel is focused for destructive actions. */
export function ConfirmDialog({ open, title, children, confirmLabel, tone = "primary", busy, error, onConfirm, onCancel }: Props) {
  const cancelRef = useRef<HTMLButtonElement>(null);
  const confirmRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    (tone === "danger" ? cancelRef : confirmRef).current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !busy) onCancel();
      e.stopPropagation(); // keep chart / label shortcuts quiet while the dialog is up
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open, tone, busy, onCancel]);

  if (!open) return null;
  const confirmClass =
    tone === "danger" ? "bg-red-600 hover:bg-red-700 text-white" : "bg-neutral-900 hover:bg-neutral-700 text-white";
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onMouseDown={() => !busy && onCancel()}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirm-title"
        className="w-full max-w-md rounded-xl bg-white p-6 shadow-xl"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <h2 id="confirm-title" className="text-lg font-semibold">
          {title}
        </h2>
        <div className="mt-2 space-y-2 text-sm text-neutral-600">{children}</div>
        {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
        <div className="mt-6 flex justify-end gap-2">
          <button
            ref={cancelRef}
            onClick={onCancel}
            disabled={busy}
            className="rounded-md border border-neutral-300 px-3 py-1.5 text-sm hover:bg-neutral-50 disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            ref={confirmRef}
            onClick={onConfirm}
            disabled={busy}
            className={`rounded-md px-3 py-1.5 text-sm font-medium disabled:opacity-50 ${confirmClass}`}
          >
            {busy ? "Working…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
