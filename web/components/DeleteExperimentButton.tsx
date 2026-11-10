"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { ConfirmDialog } from "./ConfirmDialog";

export function DeleteExperimentButton({ sessionId, name }: { sessionId: number; name: string }) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function remove() {
    setBusy(true);
    setError(null);
    const res = await fetch(`/api/sessions/${sessionId}`, { method: "DELETE" });
    setBusy(false);
    if (!res.ok) {
      setError((await res.json().catch(() => null))?.error ?? "Delete failed");
      return;
    }
    router.replace("/experiments");
    router.refresh();
  }

  return (
    <>
      <button
        onClick={() => {
          setError(null);
          setOpen(true);
        }}
        className="rounded-md border border-red-200 px-2.5 py-1 text-sm text-red-700 hover:bg-red-50"
      >
        Delete experiment
      </button>
      <ConfirmDialog
        open={open}
        title={`Delete experiment #${sessionId}?`}
        confirmLabel="Delete"
        tone="danger"
        busy={busy}
        error={error}
        onConfirm={remove}
        onCancel={() => setOpen(false)}
      >
        <p>
          <span className="font-medium text-neutral-900">{name}</span> will disappear from the experiments table and can no
          longer be opened or labelled here.
        </p>
        <p>The recording and its labels are kept in the database, so it can be restored later if needed.</p>
      </ConfirmDialog>
    </>
  );
}
