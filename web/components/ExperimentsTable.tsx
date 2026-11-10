"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { deviceName, formatBytes, formatDateTime, formatDuration } from "@/lib/format";
import type { SessionListItem } from "@/lib/types";

type SortKey = "id" | "username" | "start_time" | "duration_sec" | "device" | "total_bytes" | "status";

const COLUMNS: { key: SortKey | null; label: string; className?: string }[] = [
  { key: "id", label: "ID" },
  { key: "username", label: "User" },
  { key: "start_time", label: "Start" },
  { key: "duration_sec", label: "Duration" },
  { key: "device", label: "Device" },
  { key: null, label: "Sensors" },
  { key: "total_bytes", label: "Size", className: "text-right" },
  { key: null, label: "Note" },
  { key: null, label: "Stop" },
  { key: "status", label: "Status" },
];

const SHORT_SENSOR: Record<string, string> = {
  barometer: "baro",
  accel: "acc",
  gyro: "gyro",
  magnetic: "mag",
  orientation: "ori",
};

function sortValue(s: SessionListItem, key: SortKey): string | number {
  if (key === "device") return deviceName(s);
  if (key === "start_time") return s.start_time ?? "";
  // Approved first, then labelled-in-progress, then untouched.
  if (key === "status") return (s.approved_at ? 2_000_000 : 0) + s.label_count;
  return (s[key] as string | number | null) ?? 0;
}

export function ExperimentsTable({ sessions }: { sessions: SessionListItem[] }) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: "start_time", dir: -1 });

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = q
      ? sessions.filter((s) =>
          [s.id, s.username, s.session_tag, deviceName(s), s.recording_settings?.user_info, s.stop_reason]
            .join(" ")
            .toLowerCase()
            .includes(q),
        )
      : sessions;
    return [...filtered].sort((a, b) => {
      const va = sortValue(a, sort.key);
      const vb = sortValue(b, sort.key);
      return (va < vb ? -1 : va > vb ? 1 : 0) * sort.dir;
    });
  }, [sessions, query, sort]);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <input
          className="w-full max-w-xs rounded-md border border-neutral-300 bg-white px-3 py-1.5 text-sm outline-none focus:border-blue-500"
          placeholder="Filter by user, device, note…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <span className="text-sm text-neutral-500">
          {rows.length} of {sessions.length} sessions
        </span>
      </div>
      <div className="overflow-x-auto rounded-lg border border-neutral-200 bg-white">
        <table className="w-full text-sm">
          <thead className="bg-neutral-50 text-left text-xs uppercase tracking-wide text-neutral-500">
            <tr>
              {COLUMNS.map((c) => (
                <th key={c.label} className={`whitespace-nowrap px-3 py-2 font-medium ${c.className ?? ""}`}>
                  {c.key ? (
                    <button
                      className="uppercase tracking-wide hover:text-neutral-900"
                      onClick={() =>
                        setSort((s) => ({ key: c.key!, dir: s.key === c.key ? ((-s.dir) as 1 | -1) : 1 }))
                      }
                    >
                      {c.label}
                      {sort.key === c.key ? (sort.dir === 1 ? " ▲" : " ▼") : ""}
                    </button>
                  ) : (
                    c.label
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-100">
            {rows.map((s) => (
              <tr
                key={s.id}
                onClick={() => router.push(`/experiments/${s.id}`)}
                className="cursor-pointer hover:bg-blue-50"
              >
                <td className="px-3 py-2 tabular-nums text-neutral-500">{s.id}</td>
                <td className="px-3 py-2 font-medium">{s.username}</td>
                <td className="whitespace-nowrap px-3 py-2 tabular-nums">
                  {formatDateTime(s.start_time, s.recording_settings?.timezone)}
                </td>
                <td className="whitespace-nowrap px-3 py-2 tabular-nums">{formatDuration(s.duration_sec)}</td>
                <td className="whitespace-nowrap px-3 py-2">{deviceName(s)}</td>
                <td className="px-3 py-2">
                  <div className="flex gap-1">
                    {Object.keys(s.files ?? {})
                      .filter((k) => k in SHORT_SENSOR)
                      .map((k) => (
                        <span key={k} className="rounded bg-neutral-100 px-1.5 py-0.5 text-xs text-neutral-600">
                          {SHORT_SENSOR[k]}
                        </span>
                      ))}
                  </div>
                </td>
                <td className="whitespace-nowrap px-3 py-2 text-right tabular-nums">{formatBytes(s.total_bytes)}</td>
                <td dir="auto" title={s.recording_settings?.user_info ?? ""} className="max-w-56 truncate px-3 py-2 text-neutral-600">
                  {s.recording_settings?.user_info ?? ""}
                </td>
                <td className="whitespace-nowrap px-3 py-2 text-neutral-500">{s.stop_reason ?? "—"}</td>
                <td className="whitespace-nowrap px-3 py-2">
                  {s.approved_at ? (
                    <span className="rounded bg-green-700 px-1.5 py-0.5 text-xs font-medium text-white">✓ approved</span>
                  ) : s.label_count ? (
                    <span className="rounded bg-amber-100 px-1.5 py-0.5 text-xs text-amber-800">labelling</span>
                  ) : (
                    <span className="text-xs text-neutral-400">not labelled</span>
                  )}
                  {s.label_count > 0 && <span className="ml-1.5 text-xs text-neutral-500">{s.label_count} rides</span>}
                </td>
              </tr>
            ))}
            {!rows.length && (
              <tr>
                <td colSpan={COLUMNS.length} className="px-3 py-8 text-center text-neutral-500">
                  No sessions
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
