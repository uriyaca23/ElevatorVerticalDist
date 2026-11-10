"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { heightDiff } from "@/lib/barometer";
import { SNR_DEFAULT_BETA, SNR_MIN_SEGMENT, SNR_SIGMA_DBHZ, snrChangepoints } from "@/lib/changepoint";
import { formatClock, formatDateTime } from "@/lib/format";
import type { Label, SensorPayload, SessionReview } from "@/lib/types";
import { ConfirmDialog } from "./ConfirmDialog";
import { GpsMap } from "./GpsMap";
import { SensorCharts, type Mode, type View } from "./SensorCharts";

/** Changepoint penalty multipliers offered in the NMEA panel (higher = fewer changepoints). */
const BETAS = [1, 2, 5, 10, 20, 50];

interface Props {
  sessionId: number;
  timeZone: string;
}

const snapshot = (labels: Label[]) =>
  JSON.stringify(labels.map(({ start_ms, end_ms, type }) => [start_ms, end_ms, type]));

const sortLabels = (labels: Label[]) => [...labels].sort((a, b) => a.start_ms - b.start_ms);

export function ExperimentWorkspace({ sessionId, timeZone }: Props) {
  const [payload, setPayload] = useState<SensorPayload | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [labels, setLabels] = useState<Label[]>([]);
  const [saved, setSaved] = useState(snapshot([]));
  const [writable, setWritable] = useState(false);
  const [labelsError, setLabelsError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [mode, setMode] = useState<Mode>("zoom");
  const [selected, setSelected] = useState<number | null>(null);
  const [showPredictions, setShowPredictions] = useState(true);
  const [view, setView] = useState<View | null>(null);
  const [review, setReview] = useState<SessionReview | null>(null);
  const [reviewWritable, setReviewWritable] = useState(false);
  const [dialog, setDialog] = useState<"approve" | "reopen" | "copy" | null>(null);
  const [dialogBusy, setDialogBusy] = useState(false);
  const [dialogError, setDialogError] = useState<string | null>(null);
  const [beta, setBeta] = useState(SNR_DEFAULT_BETA);
  const [showMap, setShowMap] = useState(true);
  // Chart hover → map marker, without re-rendering the workspace on every mouse move.
  const cursorRef = useRef<(ms: number | null) => void>(() => {});
  const onCursor = useCallback((ms: number | null) => cursorRef.current(ms), []);

  useEffect(() => {
    fetch(`/api/sessions/${sessionId}/sensors`)
      .then(async (r) => {
        const body = await r.json();
        if (!r.ok) throw new Error(body.error ?? r.statusText);
        setPayload(body);
      })
      .catch((e) => setLoadError(String(e.message ?? e)));
    fetch(`/api/sessions/${sessionId}/labels`, { cache: "no-store" })
      .then((r) => r.json())
      .then((body: { labels: Label[]; writable: boolean; error?: string }) => {
        setLabels(body.labels);
        setSaved(snapshot(body.labels));
        setWritable(body.writable && !body.error);
        setLabelsError(body.error ?? null);
      });
    fetch(`/api/sessions/${sessionId}/review`, { cache: "no-store" })
      .then((r) => r.json())
      .then((body: { review: SessionReview; writable: boolean }) => {
        setReview(body.review);
        setReviewWritable(body.writable);
      });
  }, [sessionId]);

  const dirty = snapshot(labels) !== saved;
  // Approved experiments are frozen research data: labels are read-only until reopened.
  const locked = Boolean(review?.approved_at);
  const effectiveMode: Mode = locked ? "zoom" : mode;

  useEffect(() => {
    if (!dirty) return;
    const warn = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  // NMEA mean-C/N0 changepoints; cheap enough (PELT, ~1 Hz) to redo in the browser per β.
  const snrCp = useMemo(
    () => (payload?.nmea ? snrChangepoints(payload.nmea.t.map((t) => payload.t0_ms + t * 1000), payload.nmea.snr, beta) : null),
    [payload, beta],
  );

  // Barometer timestamps (ms) for Δh lookups on the smoothed height.
  const baroMs = useMemo(
    () => (payload?.barometer ? payload.barometer.t.map((t) => payload.t0_ms + t * 1000) : null),
    [payload],
  );
  const dh = useCallback(
    (s: number, e: number) =>
      baroMs && payload?.barometer ? Math.round(heightDiff(baroMs, payload.barometer.height_smooth, s, e) * 100) / 100 : null,
    [baroMs, payload],
  );

  const onCreate = useCallback(
    (start_ms: number, end_ms: number) => {
      const h = dh(start_ms, end_ms);
      // Guess the direction from the barometer; one click flips it.
      const label: Label = { start_ms, end_ms, type: h != null && h < 0 ? "down" : "up", height_diff_m: h };
      const next = sortLabels([...labels, label]);
      setLabels(next);
      setSelected(next.indexOf(label));
    },
    [dh, labels],
  );

  const onUpdate = useCallback(
    (i: number, patch: Partial<Pick<Label, "start_ms" | "end_ms">>) =>
      setLabels((ls) =>
        ls.map((l, j) => {
          if (j !== i) return l;
          const n = { ...l, ...patch };
          return { ...n, height_diff_m: dh(n.start_ms, n.end_ms) };
        }),
      ),
    [dh],
  );

  const setType = useCallback(
    (i: number, type: Label["type"]) => setLabels((ls) => ls.map((l, j) => (j === i ? { ...l, type } : l))),
    [],
  );

  const remove = useCallback((i: number) => {
    setLabels((ls) => ls.filter((_, j) => j !== i));
    setSelected(null);
  }, []);

  // Keyboard: L/Z switch mode, Delete removes, U/D set direction, Esc deselects.
  // Chart navigation keys are handled in SensorCharts.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target;
      if (t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement || t instanceof HTMLSelectElement) return;
      if (e.metaKey || e.ctrlKey) return;
      if (e.key === "Escape") setSelected(null);
      if (locked) return;
      if (e.key === "l") setMode("label");
      if (e.key === "z") setMode("zoom");
      if (selected == null) return;
      if (e.key === "Delete" || e.key === "Backspace") remove(selected);
      else if (e.key === "u") setType(selected, "up");
      else if (e.key === "d") setType(selected, "down");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected, remove, setType, locked]);

  function copyPredictions() {
    if (!payload) return;
    setLabels(payload.segments.map((s) => ({ ...s, height_diff_m: Math.round(s.height_diff_m * 100) / 100 })));
    setSelected(null);
    setDialog(null);
  }

  async function setApproved(approved: boolean) {
    setDialogBusy(true);
    setDialogError(null);
    const res = await fetch(`/api/sessions/${sessionId}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approved }),
    });
    const body = await res.json().catch(() => ({}));
    setDialogBusy(false);
    if (!res.ok) {
      setDialogError(body.error ?? "Failed");
      return;
    }
    setReview(body.review);
    if (approved) {
      setMode("zoom");
      setSelected(null);
    }
    setDialog(null);
  }

  async function save() {
    setSaving(true);
    setSaveMsg(null);
    const res = await fetch(`/api/sessions/${sessionId}/labels`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ labels }),
    });
    const body = await res.json().catch(() => ({}));
    setSaving(false);
    if (!res.ok) {
      setSaveMsg({ ok: false, text: body.error ?? "Save failed" });
      return;
    }
    setLabels(body.labels);
    setSaved(snapshot(body.labels));
    setSaveMsg({ ok: true, text: "Saved" });
  }

  const zoomTo = useCallback(
    (l: { start_ms: number; end_ms: number }) => {
      if (!payload) return;
      const a = (l.start_ms - payload.t0_ms) / 1000;
      const b = (l.end_ms - payload.t0_ms) / 1000;
      const pad = Math.max((b - a) * 0.75, 5);
      setView({ min: a - pad, max: b + pad });
    },
    [payload],
  );

  if (loadError) {
    return <section className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">Could not load sensors: {loadError}</section>;
  }
  if (!payload) {
    return <section className="rounded-lg border border-neutral-200 bg-white p-4 text-sm text-neutral-500">Downloading sensor files…</section>;
  }

  const btn = "rounded-md border px-2.5 py-1 text-sm disabled:opacity-50";
  const gps = payload.gps;
  return (
    <>
    {gps && showMap && (
      <GpsMap payload={{ ...payload, gps }} timeZone={timeZone} view={view} cursorRef={cursorRef} onZoomTo={zoomTo} />
    )}
    <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_300px]">
      <div className="min-w-0 rounded-lg border border-neutral-200 bg-white p-3">
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <div className="inline-flex overflow-hidden rounded-md border border-neutral-300 text-sm">
            {(["zoom", "label"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                disabled={locked && m === "label"}
                title={locked && m === "label" ? "Approved experiments are read-only — reopen to edit labels" : undefined}
                className={`px-3 py-1 disabled:cursor-not-allowed disabled:opacity-40 ${effectiveMode === m ? "bg-neutral-900 text-white" : "hover:bg-neutral-50"}`}
              >
                {m === "zoom" ? "Zoom (Z)" : "Label (L)"}
              </button>
            ))}
          </div>
          <button className={`${btn} border-neutral-300 hover:bg-neutral-50`} onClick={() => setView(null)} disabled={!view}>
            Show all (0)
          </button>
          <label className="ml-1 flex items-center gap-1.5 text-sm">
            <input type="checkbox" checked={showPredictions} onChange={(e) => setShowPredictions(e.target.checked)} />
            Predicted rides ({payload.segments.length})
          </label>
          {gps && (
            <label className="ml-1 flex items-center gap-1.5 text-sm">
              <input type="checkbox" checked={showMap} onChange={(e) => setShowMap(e.target.checked)} />
              GPS map
            </label>
          )}
          <span className="ml-auto text-xs text-neutral-500">
            {effectiveMode === "zoom"
              ? "Drag to zoom in on a time range"
              : "Drag to add a ride · drag its edges to adjust · click to select"}
          </span>
        </div>
        <Legend />
        <Shortcuts />
        <SensorCharts
          payload={payload}
          labels={labels}
          segments={payload.segments}
          showPredictions={showPredictions}
          selected={selected}
          mode={effectiveMode}
          timeZone={timeZone}
          view={view}
          onViewChange={setView}
          onCreate={onCreate}
          onUpdate={onUpdate}
          onSelect={setSelected}
          snrCp={snrCp}
          onCursor={onCursor}
          panelExtras={{
            nmea: snrCp && (
              <>
                <label
                  className="flex items-center gap-1"
                  title={`PELT on the per-second mean C/N0, Gaussian mean-shift cost, penalty β·σ²·ln n with σ = ${SNR_SIGMA_DBHZ} dB-Hz, segments ≥ ${SNR_MIN_SEGMENT} s. Higher β → fewer changepoints. Seconds whose GSV tracked no satellite count as 0 dB-Hz.`}
                >
                  penalty β
                  <select
                    value={beta}
                    onChange={(e) => setBeta(Number(e.target.value))}
                    className="rounded border border-neutral-300 bg-white px-1 py-0.5 text-xs"
                  >
                    {BETAS.map((b) => (
                      <option key={b} value={b}>
                        {b}
                        {b === SNR_DEFAULT_BETA ? " (default)" : ""}
                      </option>
                    ))}
                  </select>
                </label>
                <span className="tabular-nums text-neutral-500">
                  {snrCp.segments.length - 1} changepoint{snrCp.segments.length === 2 ? "" : "s"}
                </span>
              </>
            ),
          }}
        />
        {Object.entries(payload.decimated).map(([k, d]) => (
          <p key={k} className="mt-1 text-xs text-neutral-400">
            {k}: showing {d.to.toLocaleString()} of {d.from.toLocaleString()} samples (min/max decimated)
          </p>
        ))}
      </div>

      <aside className="space-y-3 rounded-lg border border-neutral-200 bg-white p-3 text-sm">
        <ApprovalCard
          review={review}
          writable={reviewWritable}
          dirty={dirty}
          timeZone={timeZone}
          onApprove={() => {
            setDialogError(null);
            setDialog("approve");
          }}
          onReopen={() => {
            setDialogError(null);
            setDialog("reopen");
          }}
        />
        <div className="flex items-center gap-2">
          <h2 className="font-semibold">Labels</h2>
          {dirty && <span className="rounded bg-amber-100 px-1.5 py-0.5 text-xs text-amber-800">unsaved</span>}
          {locked && <span className="rounded bg-neutral-100 px-1.5 py-0.5 text-xs text-neutral-600">read-only</span>}
        </div>
        <div className={`flex flex-wrap gap-2 ${locked ? "hidden" : ""}`}>
          <button className={`${btn} border-neutral-900 bg-neutral-900 text-white hover:bg-neutral-700`} disabled={!dirty || saving || !writable} onClick={save}>
            {saving ? "Saving…" : "Save"}
          </button>
          <button
            className={`${btn} border-neutral-300 hover:bg-neutral-50`}
            disabled={!payload.segments.length}
            onClick={() => (labels.length ? setDialog("copy") : copyPredictions())}
          >
            Copy predictions
          </button>
          <button
            className={`${btn} border-neutral-300 hover:bg-neutral-50`}
            disabled={!dirty}
            onClick={() => {
              const restored = JSON.parse(saved) as [number, number, Label["type"]][];
              setLabels(restored.map(([start_ms, end_ms, type]) => ({ start_ms, end_ms, type, height_diff_m: dh(start_ms, end_ms) })));
              setSelected(null);
            }}
          >
            Discard
          </button>
        </div>
        {saveMsg && <p className={saveMsg.ok ? "text-green-700" : "text-red-600"}>{saveMsg.text}</p>}
        {!writable && (
          <p className="rounded bg-amber-50 p-2 text-xs text-amber-800">
            {labelsError ? `Labels unavailable: ${labelsError}. ` : ""}
            Saving is disabled until the shaft_labels migration is applied and SUPABASE_SERVICE_ROLE_KEY is set.
          </p>
        )}

        <ol className="max-h-[520px] space-y-1 overflow-y-auto">
          {labels.map((l, i) => (
            <li
              key={`${l.start_ms}-${i}`}
              onClick={() => {
                setSelected(i);
                zoomTo(l);
              }}
              className={`flex cursor-pointer items-center gap-2 rounded border px-2 py-1.5 ${
                i === selected ? "border-blue-500 bg-blue-50" : "border-neutral-200 hover:bg-neutral-50"
              }`}
            >
              <button
                title="Flip direction (U / D)"
                disabled={locked}
                onClick={(e) => {
                  e.stopPropagation();
                  setType(i, l.type === "up" ? "down" : "up");
                }}
                className={`w-12 rounded disabled:cursor-default px-1 py-0.5 text-xs font-medium text-white ${l.type === "up" ? "bg-[#2ca02c]" : "bg-[#d62728]"}`}
              >
                {l.type === "up" ? "▲ up" : "▼ down"}
              </button>
              <div className="min-w-0 flex-1 tabular-nums">
                <div className="text-xs">
                  {formatClock(l.start_ms, timeZone).slice(0, 10)} · {((l.end_ms - l.start_ms) / 1000).toFixed(1)}s
                </div>
                <div className="text-xs text-neutral-500">Δh {l.height_diff_m == null ? "—" : `${l.height_diff_m.toFixed(2)} m`}</div>
              </div>
              <button
                title="Delete (Del)"
                onClick={(e) => {
                  e.stopPropagation();
                  remove(i);
                }}
                className={`px-1 text-neutral-400 hover:text-red-600 ${locked ? "hidden" : ""}`}
              >
                ✕
              </button>
            </li>
          ))}
          {!labels.length && (
            <li className="text-xs text-neutral-500">
              No labels yet. Switch to <b>Label</b> mode and drag across a ride, or copy the predictions and fix them.
            </li>
          )}
        </ol>

        {payload.segments.length > 0 && (
          <details className="text-xs">
            <summary className="cursor-pointer text-neutral-500">Predicted rides</summary>
            <ol className="mt-1 space-y-0.5 tabular-nums">
              {payload.segments.map((s, i) => (
                <li key={i} className="cursor-pointer hover:underline" onClick={() => zoomTo(s)}>
                  {s.type === "up" ? "▲" : "▼"} {formatClock(s.start_ms, timeZone).slice(0, 8)} ·{" "}
                  {((s.end_ms - s.start_ms) / 1000).toFixed(1)}s · Δh {s.height_diff_m.toFixed(2)} m
                </li>
              ))}
            </ol>
          </details>
        )}
      </aside>

      <ConfirmDialog
        open={dialog === "approve"}
        title="Approve for research?"
        confirmLabel="Approve"
        busy={dialogBusy}
        error={dialogError}
        onConfirm={() => setApproved(true)}
        onCancel={() => setDialog(null)}
      >
        <p>
          {labels.length
            ? `${labels.length} labelled ride${labels.length === 1 ? "" : "s"} (${labels.filter((l) => l.type === "up").length} up, ${labels.filter((l) => l.type === "down").length} down) will be marked as final ground truth.`
            : "No rides are labelled. Approve only if this recording really has no elevator rides."}
        </p>
        <p>The labels become read-only. You can reopen the experiment later to change them.</p>
      </ConfirmDialog>
      <ConfirmDialog
        open={dialog === "reopen"}
        title="Reopen for editing?"
        confirmLabel="Reopen"
        busy={dialogBusy}
        error={dialogError}
        onConfirm={() => setApproved(false)}
        onCancel={() => setDialog(null)}
      >
        <p>The experiment will no longer be marked as approved for research until it is approved again.</p>
      </ConfirmDialog>
      <ConfirmDialog
        open={dialog === "copy"}
        title="Replace labels with predictions?"
        confirmLabel="Replace"
        tone="danger"
        onConfirm={copyPredictions}
        onCancel={() => setDialog(null)}
      >
        <p>
          Your {labels.length} current label{labels.length === 1 ? "" : "s"} will be replaced by the {payload.segments.length} barometer
          predictions. Nothing is saved until you press Save.
        </p>
      </ConfirmDialog>
    </section>
    </>
  );
}

function ApprovalCard(props: {
  review: SessionReview | null;
  writable: boolean;
  dirty: boolean;
  timeZone: string;
  onApprove: () => void;
  onReopen: () => void;
}) {
  const { review, writable, dirty } = props;
  if (!review) return null;
  if (review.approved_at) {
    return (
      <div className="rounded-md border border-green-200 bg-green-50 p-2.5">
        <div className="flex items-center gap-2 font-medium text-green-800">✓ Approved for research</div>
        <p className="mt-0.5 text-xs text-green-800/80">
          by {review.approved_by ?? "unknown"} · {formatDateTime(review.approved_at, props.timeZone)}
        </p>
        <button
          onClick={props.onReopen}
          disabled={!writable}
          className="mt-2 rounded-md border border-green-300 bg-white px-2.5 py-1 text-xs text-green-800 hover:bg-green-100 disabled:opacity-50"
        >
          Reopen for editing
        </button>
      </div>
    );
  }
  return (
    <div className="rounded-md border border-neutral-200 p-2.5">
      <button
        onClick={props.onApprove}
        disabled={!writable || dirty}
        title={!writable ? "Needs the shaft_session_review migration and SUPABASE_SERVICE_ROLE_KEY" : dirty ? "Save or discard your changes first" : undefined}
        className="w-full rounded-md bg-green-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-green-800 disabled:opacity-40"
      >
        ✓ Approve for research
      </button>
      <p className="mt-1 text-xs text-neutral-500">
        {dirty ? "Save your label changes before approving." : "Mark labelling as finished once every ride is labelled."}
      </p>
    </div>
  );
}

const SHORTCUTS: [string, string][] = [
  ["← / →", "move back / forward in time (Shift: a whole screen)"],
  ["+ / −", "zoom time in / out"],
  ["↑ / ↓", "move the graph under the mouse up / down"],
  ["Shift + ↑ / ↓", "zoom that graph vertically"],
  ["F", "fit that graph vertically · 0 shows everything"],
  ["[ / ]", "previous / next predicted ride"],
  ["{ / }", "previous / next label"],
  ["Z / L", "zoom mode / label mode"],
  ["U / D / Del", "selected label: up / down / delete"],
  ["Wheel", "zoom time · on the y-axis or with Shift: zoom that graph vertically"],
  ["Drag axis", "y-axis moves that graph up/down · time axis pans"],
  ["Alt + drag", "pan freely inside a graph"],
];

function Shortcuts() {
  return (
    <details className="mb-2 px-1 text-xs text-neutral-600">
      <summary className="cursor-pointer text-neutral-500 hover:text-neutral-900">Keyboard &amp; mouse shortcuts</summary>
      <dl className="mt-1 grid grid-cols-1 gap-x-6 gap-y-0.5 sm:grid-cols-2">
        {SHORTCUTS.map(([k, v]) => (
          <div key={k} className="flex gap-2">
            <dt className="w-28 shrink-0 font-mono text-neutral-800">{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </details>
  );
}

function Legend() {
  return (
    <div className="mb-1 flex flex-wrap gap-3 px-1 text-xs text-neutral-600">
      <span className="flex items-center gap-1">
        <span className="inline-block h-3 w-4 rounded-sm border border-[#f5b041] bg-[repeating-linear-gradient(135deg,#f5b041_0_2px,transparent_2px_5px)]" />
        predicted (barometer)
      </span>
      <span className="flex items-center gap-1">
        <span className="inline-block h-3 w-4 rounded-sm border border-[#2ca02c] bg-[rgba(44,160,44,0.22)]" />
        labelled up
      </span>
      <span className="flex items-center gap-1">
        <span className="inline-block h-3 w-4 rounded-sm border border-[#d62728] bg-[rgba(214,39,40,0.22)]" />
        labelled down
      </span>
    </div>
  );
}
