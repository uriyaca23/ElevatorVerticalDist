"use client";

import { useEffect, useMemo, useRef, useState, type RefObject } from "react";
import type L from "leaflet";
import { formatDuration } from "@/lib/format";
import type { SensorPayload, StaySegment } from "@/lib/types";
import { INK, INK_MUTED, STAY_COLOR, STAY_LABEL } from "./gnssStyle";

/** A pause between fixes longer than this is not drawn as travelled. */
const BREAK_GAP_S = 300;
/** The hover marker hides when the nearest fix is further than this from the cursor time. */
const CURSOR_MAX_GAP_S = 30;

interface Props {
  payload: SensorPayload & { gps: NonNullable<SensorPayload["gps"]> };
  timeZone: string;
  /** Visible time window of the charts, seconds since t0 (null = everything). */
  view: { min: number; max: number } | null;
  /** The charts call this with the hovered time (epoch ms) or null. */
  cursorRef: RefObject<(ms: number | null) => void>;
  onZoomTo: (span: { start_ms: number; end_ms: number }) => void;
}

/** Index of the fix nearest to `t` (seconds), by binary search on ascending `ts`. */
function nearest(ts: number[], t: number): number {
  let lo = 0;
  let hi = ts.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] < t) lo = mid;
    else hi = mid;
  }
  return Math.abs(ts[lo] - t) <= Math.abs(ts[hi] - t) ? lo : hi;
}

/** Split an index range into runs with no gap longer than BREAK_GAP_S. */
function runs(ts: number[], from: number, to: number): number[][] {
  const out: number[][] = [];
  let cur: number[] = [];
  for (let i = from; i <= to; i++) {
    if (cur.length && ts[i] - ts[i - 1] > BREAK_GAP_S) {
      out.push(cur);
      cur = [];
    }
    cur.push(i);
  }
  if (cur.length) out.push(cur);
  return out;
}

const clock = (ms: number, timeZone: string) =>
  new Date(ms).toLocaleTimeString("en-GB", { timeZone, hour12: false });

export function GpsMap({ payload, timeZone, view, cursorRef, onZoomTo }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const api = useRef<{
    fitAll: () => void;
    fitSegment: (k: number) => void;
    setWindow: (v: Props["view"]) => void;
  } | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const { gps, stays, t0_ms: t0 } = payload;
  const viewRef = useRef(view);
  useEffect(() => {
    viewRef.current = view;
    api.current?.setWindow(view);
  }, [view]);
  // Latest callback for Leaflet click handlers bound once per payload.
  const onZoomRef = useRef(onZoomTo);
  useEffect(() => {
    onZoomRef.current = onZoomTo;
  });

  useEffect(() => {
    let disposed = false;
    let map: L.Map | null = null;
    let resize: ResizeObserver | null = null;
    const root = el.current;

    import("leaflet").then(({ default: Lf }) => {
      if (disposed || !root) return;
      const m = Lf.map(root, { preferCanvas: true, zoomSnap: 0.25 });
      map = m;
      const osm = Lf.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "© OpenStreetMap contributors",
      }).addTo(m);
      const sat = Lf.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
        maxZoom: 19,
        attribution: "Imagery © Esri",
      });
      Lf.control.layers({ Map: osm, Satellite: sat }, undefined, { position: "topright" }).addTo(m);
      Lf.control.scale({ imperial: false }).addTo(m);

      const ll = (i: number): L.LatLngTuple => [gps.lat[i], gps.lon[i]];
      const line = (idx: number[][]) => idx.map((r) => r.map(ll));
      const n = gps.t.length;
      // Leaflet only attaches layers once the map has a view.
      const all = Lf.latLngBounds(Array.from({ length: n }, (_, i) => ll(i)));
      const fitAll = () => m.fitBounds(all, { padding: [24, 24], maxZoom: 18 });
      fitAll();

      // Every fix in time order, thin, under everything else.
      Lf.polyline(line(runs(gps.t, 0, n - 1)), { color: "#6b6a66", weight: 1.5, opacity: 0.55, interactive: false }).addTo(m);
      // Halo along the part of the track inside the charts' time window.
      const halo = Lf.polyline([], { color: INK, weight: 9, opacity: 0.16, interactive: false, lineCap: "round" }).addTo(m);

      const segBounds: L.LatLngBounds[] = [];
      const fitSegment = (k: number) => segBounds[k] && m.fitBounds(segBounds[k], { padding: [40, 40], maxZoom: 18 });
      const layers = stays.map((s, k) => {
        // gps_stay thins to 10 s, so the fixes between one segment's last fix
        // and the next one's first belong to neither: give them to the earlier
        // segment. Lines additionally bridge to their neighbours (back to the
        // last fix before, on to the next segment's first) so the track stays
        // continuous; stays own only their fixes — the next first fix can be
        // far away (GPS re-acquired after a gap).
        const own = (stays[k + 1]?.start_index ?? s.end_index + 1) - 1;
        const tip = `#${k + 1} ${STAY_LABEL[s.classification]} · ${clock(s.start_ms, timeZone)}–${clock(s.end_ms, timeZone)} · ${formatDuration(
          (s.end_ms - s.start_ms) / 1000,
        )}${s.classification === "unknown" ? "" : ` · p=${s.probability.toFixed(2)}`}`;
        const color = STAY_COLOR[s.classification];
        const bounds = Lf.latLngBounds(Array.from({ length: s.end_index - s.start_index + 1 }, (_, q) => ll(s.start_index + q)));
        let layer: L.FeatureGroup | L.Polyline;
        if (s.classification === "stay") {
          const g = Lf.featureGroup();
          for (let i = s.start_index; i <= own; i++) {
            Lf.circleMarker(ll(i), { radius: 2.5, stroke: false, fillColor: color, fillOpacity: 0.45 }).addTo(g);
          }
          const centre = Lf.latLng(s.center_lat, s.center_lon);
          Lf.circle(centre, { radius: s.radius_m, color, weight: 2, fillColor: color, fillOpacity: 0.12 }).addTo(g);
          Lf.circleMarker(centre, { radius: 4, color: "#fff", weight: 2, fillColor: color, fillOpacity: 1 }).addTo(g);
          bounds.extend(centre.toBounds(2 * s.radius_m));
          layer = g;
        } else {
          const from = Math.max(s.start_index - 1, 0);
          const to = Math.min(own + 1, n - 1);
          layer = Lf.polyline(line(runs(gps.t, from, to)), {
            color,
            weight: s.classification === "movement" ? 4 : 3,
            opacity: 0.95,
            dashArray: s.classification === "unknown" ? "6 6" : undefined,
          });
        }
        segBounds.push(bounds);
        layer.bindTooltip(tip, { sticky: true });
        layer.on("click", () => {
          setSelected(k);
          onZoomRef.current(s);
          fitSegment(k);
        });
        return layer;
      });
      // Lines first, stays on top so their circles are never hidden under a route.
      layers.forEach((l, k) => stays[k].classification !== "stay" && l.addTo(m));
      layers.forEach((l, k) => stays[k].classification === "stay" && l.addTo(m));

      // Hover marker: where the phone was at the time under the chart cursor.
      const accuracy = Lf.circle([0, 0], { radius: 0, color: INK, weight: 1, opacity: 0.6, fillOpacity: 0.06, interactive: false });
      const marker = Lf.circleMarker([0, 0], { radius: 7, color: INK, weight: 2.5, fillColor: "#fff", fillOpacity: 1, interactive: false });
      let shown = false;
      cursorRef.current = (ms) => {
        const hide = () => {
          if (!shown) return;
          marker.remove();
          accuracy.remove();
          shown = false;
        };
        if (ms == null || !n) return hide();
        const t = (ms - t0) / 1000;
        if (t < gps.t[0] - CURSOR_MAX_GAP_S || t > gps.t[n - 1] + CURSOR_MAX_GAP_S) return hide();
        let i = nearest(gps.t, t);
        // Inside a GPS outage (common indoors) show the last known fix, dashed.
        const stale = Math.abs(gps.t[i] - t) > CURSOR_MAX_GAP_S;
        if (stale && gps.t[i] > t && i > 0) i--;
        marker.setLatLng(ll(i)).setStyle(stale ? { color: INK_MUTED, dashArray: "3 3" } : { color: INK, dashArray: undefined });
        accuracy.setLatLng(ll(i)).setRadius(stale ? 0 : (gps.accuracy[i] ?? 0));
        if (!shown) {
          accuracy.addTo(m);
          marker.addTo(m);
          shown = true;
        }
      };

      const setWindow = (v: Props["view"]) => {
        if (!v) {
          halo.setLatLngs([]);
          return;
        }
        const a = gps.t.findIndex((t) => t >= v.min);
        let b = a;
        while (b + 1 < n && gps.t[b + 1] <= v.max) b++;
        halo.setLatLngs(a < 0 || gps.t[a] > v.max ? [] : line(runs(gps.t, a, b)));
      };
      api.current = { fitAll, fitSegment, setWindow };
      setWindow(viewRef.current);

      resize = new ResizeObserver(() => m.invalidateSize());
      resize.observe(root);
    });

    return () => {
      disposed = true;
      api.current = null;
      cursorRef.current = () => {};
      resize?.disconnect();
      map?.remove();
    };
  }, [gps, stays, t0, timeZone, cursorRef]);

  const counts = useMemo(() => {
    const c = { stay: 0, movement: 0, unknown: 0 };
    for (const s of stays) c[s.classification]++;
    return c;
  }, [stays]);

  const pick = (k: number, s: StaySegment) => {
    setSelected(k);
    onZoomTo(s);
    api.current?.fitSegment(k);
  };

  return (
    <section className="rounded-lg border border-neutral-200 bg-white p-3">
      <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1">
        <h2 className="text-sm font-semibold">GPS track</h2>
        <MapLegend />
        <span className="text-xs text-neutral-500">
          {gps.t.length.toLocaleString()} fixes (best per second) · {counts.stay} stays · {counts.movement} movements
          {counts.unknown ? ` · ${counts.unknown} unknown` : ""}
        </span>
        <button
          className="ml-auto rounded-md border border-neutral-300 px-2.5 py-1 text-xs hover:bg-neutral-50"
          onClick={() => {
            setSelected(null);
            api.current?.fitAll();
          }}
        >
          Fit track
        </button>
      </div>
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div ref={el} className="isolate h-[440px] w-full rounded-md border border-neutral-200" />
        <div className="flex min-h-0 flex-col">
          <p className="mb-1 text-xs text-neutral-500">
            gps_stay segmentation — click to zoom the map and the charts. Hover the charts to see the position.
          </p>
          <ol className="max-h-[410px] space-y-1 overflow-y-auto text-xs tabular-nums">
            {stays.map((s, k) => (
              <li
                key={k}
                onClick={() => pick(k, s)}
                className={`flex cursor-pointer items-center gap-2 rounded border px-2 py-1.5 ${
                  k === selected ? "border-blue-500 bg-blue-50" : "border-neutral-200 hover:bg-neutral-50"
                }`}
              >
                <Swatch cls={s.classification} />
                <span className="w-6 text-neutral-400">#{k + 1}</span>
                <span className="w-16 font-medium text-neutral-800">{STAY_LABEL[s.classification]}</span>
                <span className="text-neutral-700">{clock(s.start_ms, timeZone)}</span>
                <span className="text-neutral-500">{formatDuration((s.end_ms - s.start_ms) / 1000)}</span>
                <span className="ml-auto text-neutral-500">{s.classification === "unknown" ? "—" : `p ${s.probability.toFixed(2)}`}</span>
              </li>
            ))}
            {!stays.length && <li className="text-neutral-500">Too few fixes to segment.</li>}
          </ol>
        </div>
      </div>
    </section>
  );
}

function Swatch({ cls }: { cls: StaySegment["classification"] }) {
  const color = STAY_COLOR[cls];
  if (cls === "stay") {
    return <span className="inline-block h-3 w-3 shrink-0 rounded-full border-2" style={{ borderColor: color, background: `${color}33` }} />;
  }
  return (
    <span
      className="inline-block h-0 w-4 shrink-0 border-t-[3px]"
      style={{ borderColor: color, borderTopStyle: cls === "unknown" ? "dashed" : "solid" }}
    />
  );
}

function MapLegend() {
  return (
    <span className="flex flex-wrap items-center gap-3 text-xs text-neutral-600">
      <span className="flex items-center gap-1">
        <Swatch cls="stay" /> stay (50 m)
      </span>
      <span className="flex items-center gap-1">
        <Swatch cls="movement" /> movement
      </span>
      <span className="flex items-center gap-1">
        <Swatch cls="unknown" /> unknown
      </span>
      <span className="flex items-center gap-1">
        <span className="inline-block h-0 w-4 border-t-[1.5px] border-[#6b6a66]" /> all fixes
      </span>
      <span className="flex items-center gap-1">
        <span className="inline-block h-2.5 w-2.5 rounded-full border-2 border-neutral-900 bg-white" /> chart cursor
      </span>
      <span className="flex items-center gap-1">
        <span className="inline-block h-2.5 w-2.5 rounded-full border-2 border-dashed border-[#9a9993] bg-white" /> no fix (last known)
      </span>
    </span>
  );
}
