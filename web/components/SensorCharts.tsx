"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type uPlot from "uplot";
import type { SnrChangepoints } from "@/lib/changepoint";
import { formatClock } from "@/lib/format";
import type { Label, Segment, SensorPayload } from "@/lib/types";
import { INK, INK_2, INK_MUTED } from "./gnssStyle";

export type Mode = "zoom" | "label";
export interface View {
  min: number;
  max: number;
}

interface Props {
  payload: SensorPayload;
  labels: Label[];
  segments: Segment[];
  showPredictions: boolean;
  selected: number | null;
  mode: Mode;
  timeZone: string;
  view: View | null;
  onViewChange: (view: View | null) => void;
  onCreate: (startMs: number, endMs: number) => void;
  onUpdate: (index: number, patch: Partial<Pick<Label, "start_ms" | "end_ms">>) => void;
  onSelect: (index: number | null) => void;
  /** NMEA mean-C/N0 changepoints (lib/changepoint.ts), drawn in the NMEA panel. */
  snrCp?: SnrChangepoints | null;
  /** Called with the epoch ms under the mouse in any panel, or null when it leaves. */
  onCursor?: (ms: number | null) => void;
  /** Extra controls / legend in a panel's header. */
  panelExtras?: Partial<Record<PanelKey, ReactNode>>;
}

// Same palette as src/data/gt_editor.py so labels read the same in both tools.
const TYPE_FILL = { up: "rgba(44,160,44,0.22)", down: "rgba(214,39,40,0.22)" };
const TYPE_EDGE = { up: "#2ca02c", down: "#d62728" };
const PREDICTION = "#f5b041";
const SELECTED = "#1f6feb";
const AXIS_COLORS = { x: "#4269d0", y: "#ff725c", z: "#9467bd" };
const EDGE_GRAB_PX = 8;
const MIN_LABEL_PX = 4;
const X_ZOOM_STEP = 1.4;
const Y_ZOOM_STEP = 1.25;
const Y_PAN_STEP = 0.15;

export type PanelKey = "barometer" | "accel" | "gyro" | "nmea";

interface PanelSpec {
  key: PanelKey;
  title: string;
  unit: string;
  height: number;
  /** `fit` is the NMEA changepoint fit, used by the nmea panel only. */
  data: (p: SensorPayload, fit: number[] | null) => uPlot.AlignedData | null;
  series: uPlot.Series[];
}

const PANELS: PanelSpec[] = [
  {
    key: "barometer",
    title: "Barometer height",
    unit: "m",
    height: 260,
    data: (p) => (p.barometer ? [p.barometer.t, p.barometer.height, p.barometer.height_smooth] : null),
    series: [
      { label: "raw", stroke: "#c4c4c4", width: 1 },
      { label: "median-51", stroke: "#1f77b4", width: 1.5 },
    ],
  },
  {
    key: "accel",
    title: "Accelerometer",
    unit: "m/s²",
    height: 230,
    data: (p) => (p.accel ? [p.accel.t, p.accel.a_rec, p.accel.a_mag_g, p.accel.x, p.accel.y, p.accel.z] : null),
    series: [
      { label: "a_rec (Mahony)", stroke: "#111", width: 1.25 },
      { label: "|a|−g", stroke: "#9a9a9a", width: 1, show: false },
      { label: "x", stroke: AXIS_COLORS.x, width: 1, show: false },
      { label: "y", stroke: AXIS_COLORS.y, width: 1, show: false },
      { label: "z", stroke: AXIS_COLORS.z, width: 1, show: false },
    ],
  },
  {
    key: "gyro",
    title: "Gyroscope",
    unit: "rad/s",
    height: 170,
    data: (p) => (p.gyro ? [p.gyro.t, p.gyro.x, p.gyro.y, p.gyro.z] : null),
    series: [
      { label: "x", stroke: AXIS_COLORS.x, width: 1 },
      { label: "y", stroke: AXIS_COLORS.y, width: 1 },
      { label: "z", stroke: AXIS_COLORS.z, width: 1 },
    ],
  },
  {
    key: "nmea",
    title: "NMEA — mean C/N0, changepoints",
    unit: "dB-Hz",
    height: 210,
    data: (p, fit) => (p.nmea ? [p.nmea.t, p.nmea.snr, fit ?? p.nmea.snr.map(() => null)] : null),
    series: [
      { label: "mean C/N0", stroke: INK_MUTED, width: 1 },
      { label: "changepoint mean", stroke: INK, width: 2 },
    ],
  },
];

/** Imperative chart controls, shared by keyboard shortcuts and panel buttons. */
interface ChartApi {
  panX: (fraction: number) => void;
  zoomX: (factor: number) => void;
  reset: () => void;
  panY: (key: PanelKey, fraction: number) => void;
  zoomY: (key: PanelKey, factor: number) => void;
  fitY: (key: PanelKey) => void;
  jump: (direction: 1 | -1, kind: "prediction" | "label") => void;
}

/** Wall-clock tick labels in the recording's timezone (HH:MM:SS, plus .s when ticks are sub-second). */
function clockTicks(t0Ms: number, timeZone: string) {
  const fmt = new Intl.DateTimeFormat("en-GB", { timeZone, hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
  return (_u: uPlot, splits: number[]) => {
    const step = splits.length > 1 ? splits[1] - splits[0] : 1;
    return splits.map((s) => {
      const ms = t0Ms + s * 1000;
      const base = fmt.format(ms);
      return step < 1 ? `${base}.${String(Math.floor(((ms % 1000) + 1000) % 1000 / 100))}` : base;
    });
  };
}

function hatchPattern(ctx: CanvasRenderingContext2D): CanvasPattern | string {
  const c = document.createElement("canvas");
  c.width = c.height = 10;
  const g = c.getContext("2d");
  if (!g) return "rgba(245,176,65,0.25)";
  g.fillStyle = "rgba(245,176,65,0.12)";
  g.fillRect(0, 0, 10, 10);
  g.strokeStyle = PREDICTION;
  g.lineWidth = 1.5;
  g.beginPath();
  g.moveTo(-1, 11);
  g.lineTo(11, -1);
  g.stroke();
  return ctx.createPattern(c, "repeat") ?? "rgba(245,176,65,0.25)";
}

function autoRange(dmin: number | null, dmax: number | null): uPlot.Range.MinMax {
  if (dmin == null || dmax == null || !Number.isFinite(dmin) || !Number.isFinite(dmax)) return [0, 1];
  const pad = (dmax - dmin) * 0.08 || Math.abs(dmax) * 0.05 || 1;
  return [dmin - pad, dmax + pad];
}

export function SensorCharts(props: Props) {
  const container = useRef<HTMLDivElement>(null);
  const api = useRef<ChartApi | null>(null);
  const [active, setActive] = useState<PanelKey>("barometer");
  const activeRef = useRef<PanelKey>("barometer");
  const [collapsed, setCollapsed] = useState<Partial<Record<PanelKey, boolean>>>({});
  // Latest props for uPlot hooks/listeners, which are bound once per payload.
  const live = useRef(props);
  const redraw = useRef<() => void>(() => {});
  const fitAll = useRef<() => void>(() => {});
  // The last window the charts themselves set; anything else is an external jump.
  const lastSent = useRef<View | null>(null);
  const applyView = useRef<(v: View | null) => void>(() => {});
  const setNmeaFit = useRef<(fit: number[] | null) => void>(() => {});
  useEffect(() => {
    live.current = props;
  });

  const { payload } = props;
  const panelData = useMemo(
    () => Object.fromEntries(PANELS.map((spec) => [spec.key, spec.data(payload, null)])) as Record<PanelKey, uPlot.AlignedData | null>,
    [payload],
  );
  const overviewData = useMemo((): uPlot.AlignedData | null => {
    if (payload.barometer) return [payload.barometer.t, payload.barometer.height_smooth];
    if (payload.accel) return [payload.accel.t, payload.accel.a_rec];
    return null;
  }, [payload]);

  const markActive = (key: PanelKey) => {
    activeRef.current = key;
    setActive(key);
  };

  useEffect(() => {
    let disposed = false;
    const root = container.current;
    if (!root) return;
    const cleanups: (() => void)[] = [];
    const plots: Partial<Record<PanelKey, uPlot>> = {};
    let overview: uPlot | null = null;
    const yLock: Partial<Record<PanelKey, { min: number; max: number }>> = {};

    import("uplot").then(({ default: uPlotCtor }) => {
      if (disposed) return;
      const t0 = payload.t0_ms;
      const toMs = (sec: number) => t0 + sec * 1000;
      const toSec = (ms: number) => (ms - t0) / 1000;
      const full: View = { min: 0, max: Math.max((payload.t1_ms - t0) / 1000, 1) };
      const fullWidth = full.max - full.min;
      const current = () => live.current.view ?? full;
      const allPlots = () => Object.values(plots) as uPlot[];
      const ticks = clockTicks(t0, live.current.timeZone);
      let pattern: CanvasPattern | string | null = null;

      /** Keep the window inside the recording; a full-width window means "no zoom". */
      const clamp = (v: View | null): View | null => {
        if (!v) return null;
        const w = Math.max(v.max - v.min, 0.5);
        if (w >= fullWidth) return null;
        let min = v.min;
        if (min < full.min) min = full.min;
        if (min + w > full.max) min = full.max - w;
        return { min, max: min + w };
      };
      applyView.current = (v) => {
        const r = v ?? full;
        for (const u of allPlots()) {
          const s = u.scales.x;
          if (s.min !== r.min || s.max !== r.max) u.setScale("x", { min: r.min, max: r.max });
        }
        overview?.redraw(false, false);
      };
      /** Set the time window. `fit` re-fits every graph vertically (used for jumps). */
      const setView = (v: View | null, fit = false) => {
        const c = clamp(v);
        if (fit) for (const k of Object.keys(yLock) as PanelKey[]) delete yLock[k];
        applyView.current(c);
        lastSent.current = c;
        live.current.onViewChange(c);
      };
      fitAll.current = () => {
        for (const k of Object.keys(yLock) as PanelKey[]) delete yLock[k];
        applyView.current(live.current.view);
        for (const u of allPlots()) u.setScale("x", { min: u.scales.x.min!, max: u.scales.x.max! });
      };

      const setY = (key: PanelKey, min: number, max: number) => {
        const u = plots[key];
        if (!u || !(max > min)) return;
        yLock[key] = { min, max };
        u.setScale("y", { min, max });
      };
      const yOf = (key: PanelKey) => {
        const s = plots[key]?.scales.y;
        return s && s.min != null && s.max != null ? { min: s.min, max: s.max } : null;
      };

      api.current = {
        panX: (f) => {
          const v = current();
          const d = (v.max - v.min) * f;
          setView({ min: v.min + d, max: v.max + d });
        },
        zoomX: (k) => {
          const v = current();
          const c = (v.min + v.max) / 2;
          const half = ((v.max - v.min) * k) / 2;
          setView({ min: c - half, max: c + half });
        },
        reset: () => {
          setView(null, true);
          fitAll.current();
        },
        panY: (key, f) => {
          const y = yOf(key);
          if (!y) return;
          const d = (y.max - y.min) * f;
          setY(key, y.min + d, y.max + d);
        },
        zoomY: (key, k) => {
          const y = yOf(key);
          if (!y) return;
          const c = (y.min + y.max) / 2;
          setY(key, c - (c - y.min) * k, c + (y.max - c) * k);
        },
        fitY: (key) => {
          const u = plots[key];
          if (!u) return;
          delete yLock[key];
          const { min, max } = u.scales.x;
          u.setScale("x", { min: min!, max: max! }); // re-runs the auto y range
        },
        jump: (dir, kind) => {
          const p = live.current;
          const items = (kind === "prediction" ? p.segments : p.labels)
            .map((s, index) => ({ a: toSec(s.start_ms), b: toSec(s.end_ms), index }))
            .sort((x, y) => x.a - y.a);
          if (!items.length) return;
          const v = current();
          const c = (v.min + v.max) / 2;
          const mid = (it: (typeof items)[number]) => (it.a + it.b) / 2;
          const target =
            dir > 0 ? items.find((it) => mid(it) > c + 0.01) : [...items].reverse().find((it) => mid(it) < c - 0.01);
          if (!target) return;
          const pad = Math.max((target.b - target.a) * 0.75, 5);
          setView({ min: target.a - pad, max: target.b + pad }, true);
          if (kind === "label") p.onSelect(target.index);
        },
      };

      /** NMEA panel background: a dashed line at every changepoint (the segment means are the black step series). */
      function drawBackground(key: PanelKey, u: uPlot) {
        if (key !== "nmea") return;
        const { ctx, bbox } = u;
        const dpr = devicePixelRatio;
        ctx.save();
        ctx.beginPath();
        ctx.rect(bbox.left, bbox.top, bbox.width, bbox.height);
        ctx.clip();
        ctx.strokeStyle = INK_2;
        ctx.lineWidth = dpr;
        ctx.setLineDash([3 * dpr, 3 * dpr]);
        ctx.beginPath();
        for (const s of (live.current.snrCp?.segments ?? []).slice(1)) {
          const x = u.valToPos(toSec(s.start_ms), "x", true);
          if (x < bbox.left || x > bbox.left + bbox.width) continue;
          ctx.moveTo(x, bbox.top);
          ctx.lineTo(x, bbox.top + bbox.height);
        }
        ctx.stroke();
        ctx.restore();
      }

      function drawOverlays(u: uPlot) {
        const { ctx, bbox } = u;
        const p = live.current;
        pattern ??= hatchPattern(ctx);
        const xPos = (ms: number) => u.valToPos(toSec(ms), "x", true);
        const clip = (a: number, b: number): [number, number] | null => {
          const l = Math.max(a, bbox.left);
          const r = Math.min(b, bbox.left + bbox.width);
          return r > l ? [l, r] : null;
        };
        ctx.save();
        if (p.showPredictions) {
          // Predictions occupy the upper strip so labels below stay readable.
          ctx.fillStyle = pattern;
          for (const s of p.segments) {
            const c = clip(xPos(s.start_ms), xPos(s.end_ms));
            if (c) ctx.fillRect(c[0], bbox.top, c[1] - c[0], bbox.height * 0.35);
          }
        }
        p.labels.forEach((l, i) => {
          const c = clip(xPos(l.start_ms), xPos(l.end_ms));
          if (!c) return;
          ctx.fillStyle = TYPE_FILL[l.type];
          ctx.fillRect(c[0], bbox.top, c[1] - c[0], bbox.height);
          ctx.strokeStyle = i === p.selected ? SELECTED : TYPE_EDGE[l.type];
          ctx.lineWidth = (i === p.selected ? 2.5 : 1.25) * devicePixelRatio;
          ctx.beginPath();
          for (const x of [xPos(l.start_ms), xPos(l.end_ms)]) {
            if (x < bbox.left || x > bbox.left + bbox.width) continue;
            ctx.moveTo(x, bbox.top);
            ctx.lineTo(x, bbox.top + bbox.height);
          }
          ctx.stroke();
        });
        ctx.restore();
      }

      /** Label index + edge under a CSS-pixel x position in the plot area. */
      function edgeAt(u: uPlot, cssX: number): { index: number; edge: "start_ms" | "end_ms" } | null {
        let best: { index: number; edge: "start_ms" | "end_ms"; d: number } | null = null;
        live.current.labels.forEach((l, index) => {
          for (const edge of ["start_ms", "end_ms"] as const) {
            const d = Math.abs(u.valToPos(toSec(l[edge]), "x") - cssX);
            if (d <= EDGE_GRAB_PX && (!best || d < best.d)) best = { index, edge, d };
          }
        });
        return best;
      }

      function labelAt(u: uPlot, cssX: number): number | null {
        const ms = toMs(u.posToVal(cssX, "x"));
        const i = live.current.labels.findIndex((l) => ms >= l.start_ms && ms <= l.end_ms);
        return i >= 0 ? i : null;
      }

      /** Drag helper: calls `move(dx, dy)` in CSS px relative to the press point. */
      function drag(e: MouseEvent, move: (dx: number, dy: number) => void) {
        let frame = 0;
        const onMove = (ev: MouseEvent) => {
          cancelAnimationFrame(frame);
          frame = requestAnimationFrame(() => move(ev.clientX - e.clientX, ev.clientY - e.clientY));
        };
        const onUp = () => {
          cancelAnimationFrame(frame);
          window.removeEventListener("mousemove", onMove);
          window.removeEventListener("mouseup", onUp);
        };
        window.addEventListener("mousemove", onMove);
        window.addEventListener("mouseup", onUp);
      }

      const width = root.clientWidth;
      const tz = live.current.timeZone;
      // Only the panel under the mouse reports the cursor time (synced
      // panels mirror it, and a collapsed one would report garbage).
      let hovered: uPlot | null = null;
      for (const spec of PANELS) {
        const data = panelData[spec.key] && spec.data(payload, live.current.snrCp?.fit ?? null);
        const el = root.querySelector<HTMLDivElement>(`[data-panel="${spec.key}"]`);
        if (!el || !data) continue;
        const opts: uPlot.Options = {
          width,
          height: spec.height,
          cursor: {
            sync: { key: "shaft-x", setSeries: false },
            drag: { x: true, y: false, setScale: false },
          },
          scales: {
            x: { time: false, min: full.min, max: full.max },
            y: { range: (_u, dmin, dmax) => (yLock[spec.key] ? [yLock[spec.key]!.min, yLock[spec.key]!.max] : autoRange(dmin, dmax)) },
          },
          axes: [
            { values: ticks, size: 32 },
            { label: spec.unit, size: 60, labelSize: 16 },
          ],
          legend: { live: true },
          series: [{ label: "time", value: (_u, v) => (v == null ? "—" : formatClock(toMs(v), tz)) }, ...spec.series],
          hooks: {
            drawClear: [
              (u) => {
                drawBackground(spec.key, u);
                drawOverlays(u);
              },
            ],
            setCursor: [
              (u) => {
                if (u !== hovered) return;
                const l = u.cursor.left;
                live.current.onCursor?.(l == null || l < 0 ? null : toMs(u.posToVal(l, "x")));
              },
            ],
            setSelect: [
              (u) => {
                const { left, width: w } = u.select;
                if (w < MIN_LABEL_PX) return;
                const a = u.posToVal(left, "x");
                const b = u.posToVal(left + w, "x");
                u.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
                if (live.current.mode === "zoom") setView({ min: a, max: b });
                else live.current.onCreate(Math.round(toMs(a)), Math.round(toMs(b)));
              },
            ],
          },
        };
        const u = new uPlotCtor(opts, data, el);
        plots[spec.key] = u;
        if (spec.key === "nmea")
          setNmeaFit.current = (fit) => {
            // setData(…, false) keeps the zoom but does not repaint on its own.
            u.setData(spec.data(payload, fit)!, false);
            u.redraw();
          };

        // Mouse handling on the whole plot frame (plot area + axes). The
        // mousedown listener runs in the capture phase so axis drags and
        // label-edge grabs pre-empt uPlot's own drag-to-select.
        const over = u.over;
        const wrap = over.parentElement!;
        const region = (e: MouseEvent) => {
          const r = over.getBoundingClientRect();
          const x = e.clientX - r.left;
          const y = e.clientY - r.top;
          const inY = y >= 0 && y <= r.height;
          const inX = x >= 0 && x <= r.width;
          return { x, y, r, plot: inX && inY, yAxis: x < 0 && inY, xAxis: inX && y > r.height };
        };
        let downX = 0;
        const onDown = (e: MouseEvent) => {
          downX = e.clientX;
          markActive(spec.key);
          if (e.button !== 0) return;
          const g = region(e);
          const grab = () => {
            e.stopPropagation();
            e.preventDefault();
          };
          if (g.yAxis || (g.plot && e.altKey)) {
            // Drag the y axis (or Alt+drag inside the plot) to move this graph up/down.
            grab();
            const y0 = yOf(spec.key);
            const v0 = current();
            if (!y0) return;
            drag(e, (dx, dy) => {
              const d = (dy / g.r.height) * (y0.max - y0.min);
              setY(spec.key, y0.min + d, y0.max + d);
              if (e.altKey) {
                const dt = (dx / g.r.width) * (v0.max - v0.min);
                setView({ min: v0.min - dt, max: v0.max - dt });
              }
            });
            return;
          }
          if (g.xAxis) {
            // Drag the time axis to pan.
            grab();
            const v0 = current();
            drag(e, (dx) => {
              const dt = (dx / g.r.width) * (v0.max - v0.min);
              setView({ min: v0.min - dt, max: v0.max - dt });
            });
            return;
          }
          if (!g.plot || live.current.mode !== "label") return;
          const hit = edgeAt(u, g.x);
          if (!hit) return;
          grab();
          live.current.onSelect(hit.index);
          drag(e, (dx) => {
            const ms = Math.round(toMs(u.posToVal(g.x + dx, "x")));
            const l = live.current.labels[hit.index];
            if (!l) return;
            // Keep at least 100 ms between edges.
            const v = hit.edge === "start_ms" ? Math.min(ms, l.end_ms - 100) : Math.max(ms, l.start_ms + 100);
            live.current.onUpdate(hit.index, { [hit.edge]: v });
          });
        };
        const onHover = (e: MouseEvent) => {
          const g = region(e);
          const mode = live.current.mode;
          wrap.style.cursor = g.yAxis ? "ns-resize" : g.xAxis ? "ew-resize" : "";
          over.style.cursor =
            mode === "label" && edgeAt(u, g.x) ? "ew-resize" : e.altKey ? "move" : mode === "label" ? "crosshair" : "zoom-in";
        };
        const onClick = (e: MouseEvent) => {
          if (Math.abs(e.clientX - downX) > 3) return;
          live.current.onSelect(labelAt(u, e.offsetX));
        };
        const onWheel = (e: WheelEvent) => {
          const g = region(e);
          if (!g.plot && !g.yAxis && !g.xAxis) return;
          e.preventDefault();
          // macOS turns Shift+wheel into horizontal scroll; accept either delta.
          const delta = e.deltaY || e.deltaX;
          if (g.yAxis || e.shiftKey) {
            const y = yOf(spec.key);
            if (!y) return;
            const at = g.yAxis || g.plot ? u.posToVal(Math.min(Math.max(g.y, 0), g.r.height), "y") : (y.min + y.max) / 2;
            const k = delta > 0 ? Y_ZOOM_STEP : 1 / Y_ZOOM_STEP;
            setY(spec.key, at - (at - y.min) * k, at + (y.max - at) * k);
            return;
          }
          const v = current();
          const span = v.max - v.min;
          if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
            const shift = (e.deltaX / g.r.width) * span;
            setView({ min: v.min + shift, max: v.max + shift });
            return;
          }
          const at = u.posToVal(Math.min(Math.max(g.x, 0), g.r.width), "x");
          const k = e.deltaY > 0 ? 1.25 : 0.8;
          setView({ min: at - (at - v.min) * k, max: at + (v.max - at) * k });
        };
        const onDbl = (e: MouseEvent) => {
          if (region(e).yAxis) api.current?.fitY(spec.key);
          else setView(null);
        };
        const onEnter = () => {
          markActive(spec.key);
          hovered = u;
        };
        const onLeave = () => {
          hovered = null;
          live.current.onCursor?.(null);
        };
        wrap.addEventListener("mousedown", onDown, true);
        wrap.addEventListener("mousemove", onHover);
        wrap.addEventListener("wheel", onWheel, { passive: false });
        wrap.addEventListener("dblclick", onDbl);
        wrap.addEventListener("mouseenter", onEnter);
        wrap.addEventListener("mouseleave", onLeave);
        over.addEventListener("click", onClick);
        cleanups.push(() => {
          wrap.removeEventListener("mousedown", onDown, true);
          wrap.removeEventListener("mousemove", onHover);
          wrap.removeEventListener("wheel", onWheel);
          wrap.removeEventListener("dblclick", onDbl);
          wrap.removeEventListener("mouseenter", onEnter);
          wrap.removeEventListener("mouseleave", onLeave);
          over.removeEventListener("click", onClick);
        });
      }

      // Overview strip: the whole recording, with the visible window outlined.
      const ovEl = root.querySelector<HTMLDivElement>("[data-overview]");
      if (ovEl && overviewData) {
        overview = new uPlotCtor(
          {
            width,
            height: 70,
            cursor: { show: false, drag: { x: false, y: false } },
            legend: { show: false },
            scales: { x: { time: false, min: full.min, max: full.max } },
            axes: [
              { values: ticks, size: 22, font: "10px sans-serif" },
              { show: false },
            ],
            series: [{}, { stroke: "#1f77b4", width: 1 }],
            hooks: {
              drawClear: [
                (u) => {
                  const { ctx, bbox } = u;
                  const p = live.current;
                  const x = (ms: number) => u.valToPos(toSec(ms), "x", true);
                  ctx.save();
                  if (p.showPredictions) {
                    ctx.fillStyle = "rgba(245,176,65,0.55)";
                    for (const s of p.segments) ctx.fillRect(x(s.start_ms), bbox.top, Math.max(1, x(s.end_ms) - x(s.start_ms)), bbox.height * 0.3);
                  }
                  for (const l of p.labels) {
                    ctx.fillStyle = TYPE_EDGE[l.type];
                    ctx.fillRect(x(l.start_ms), bbox.top + bbox.height * 0.75, Math.max(1, x(l.end_ms) - x(l.start_ms)), bbox.height * 0.25);
                  }
                  ctx.restore();
                },
              ],
              draw: [
                (u) => {
                  const v = live.current.view;
                  if (!v) return;
                  const { ctx, bbox } = u;
                  const a = u.valToPos(v.min, "x", true);
                  const b = u.valToPos(v.max, "x", true);
                  ctx.save();
                  ctx.fillStyle = "rgba(0,0,0,0.12)";
                  ctx.fillRect(bbox.left, bbox.top, a - bbox.left, bbox.height);
                  ctx.fillRect(b, bbox.top, bbox.left + bbox.width - b, bbox.height);
                  ctx.strokeStyle = SELECTED;
                  ctx.lineWidth = 1.5 * devicePixelRatio;
                  ctx.strokeRect(a, bbox.top, Math.max(b - a, 2), bbox.height);
                  ctx.restore();
                },
              ],
            },
          },
          overviewData,
          ovEl,
        );
        const ov = overview;
        const onOvDown = (e: MouseEvent) => {
          if (e.button !== 0) return;
          e.preventDefault();
          const r = ov.over.getBoundingClientRect();
          const at = (clientX: number) => ov.posToVal(Math.min(Math.max(clientX - r.left, 0), r.width), "x");
          const a = at(e.clientX);
          const v = current();
          const w = v.max - v.min;
          const inside = live.current.view && a >= v.min && a <= v.max;
          drag(e, (dx) => {
            if (inside) {
              // Grab the window and slide it.
              const dt = (dx / r.width) * fullWidth;
              setView({ min: v.min + dt, max: v.max + dt });
            } else if (Math.abs(dx) > 4) {
              // Sweep out a new window.
              const b = at(e.clientX + dx);
              setView({ min: Math.min(a, b), max: Math.max(a, b) });
            }
          });
          const onUp = (ev: MouseEvent) => {
            window.removeEventListener("mouseup", onUp);
            if (!inside && Math.abs(ev.clientX - e.clientX) <= 4) {
              // Click: centre the current window width there.
              const ww = live.current.view ? w : Math.min(60, fullWidth / 4);
              setView({ min: a - ww / 2, max: a + ww / 2 }, true);
            }
          };
          window.addEventListener("mouseup", onUp);
        };
        ov.over.style.cursor = "pointer";
        ov.over.addEventListener("mousedown", onOvDown);
        cleanups.push(() => ov.over.removeEventListener("mousedown", onOvDown));
      }

      redraw.current = () => {
        for (const u of allPlots()) u.redraw(false, false);
        overview?.redraw(false, false);
      };
      applyView.current(live.current.view);

      const resize = new ResizeObserver(() => {
        for (const u of [...allPlots(), ...(overview ? [overview] : [])]) u.setSize({ width: root.clientWidth, height: u.height });
      });
      resize.observe(root);
      cleanups.push(() => resize.disconnect());
    });

    return () => {
      disposed = true;
      api.current = null;
      redraw.current = () => {};
      fitAll.current = () => {};
      applyView.current = () => {};
      setNmeaFit.current = () => {};
      cleanups.forEach((f) => f());
      Object.values(plots).forEach((u) => u.destroy());
      overview?.destroy();
    };
  }, [panelData, overviewData, payload]);

  // Redraw overlays when labels / selection / toggles change.
  useEffect(() => {
    redraw.current();
  }, [props.labels, props.selected, props.showPredictions, props.segments]);

  // New changepoints (sensitivity changed): swap the NMEA fit without rebuilding the charts.
  useEffect(() => {
    setNmeaFit.current(props.snrCp?.fit ?? null);
  }, [props.snrCp]);

  // Apply externally-driven zoom (label list, "Show all"); those are jumps, so re-fit vertically.
  useEffect(() => {
    if (props.view === lastSent.current) return;
    lastSent.current = props.view;
    fitAll.current();
  }, [props.view]);

  // Keyboard navigation. Label editing keys live in ExperimentWorkspace.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target;
      if (t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement || t instanceof HTMLSelectElement) return;
      if (e.metaKey || e.ctrlKey) return;
      const a = api.current;
      if (!a) return;
      const key = activeRef.current;
      switch (e.key) {
        case "ArrowLeft":
          a.panX(e.shiftKey ? -0.9 : -0.2);
          break;
        case "ArrowRight":
          a.panX(e.shiftKey ? 0.9 : 0.2);
          break;
        case "ArrowUp":
          if (e.shiftKey) a.zoomY(key, 1 / Y_ZOOM_STEP);
          else a.panY(key, Y_PAN_STEP);
          break;
        case "ArrowDown":
          if (e.shiftKey) a.zoomY(key, Y_ZOOM_STEP);
          else a.panY(key, -Y_PAN_STEP);
          break;
        case "+":
        case "=":
          a.zoomX(1 / X_ZOOM_STEP);
          break;
        case "-":
        case "_":
          a.zoomX(X_ZOOM_STEP);
          break;
        case "0":
          a.reset();
          break;
        case "f":
          a.fitY(key);
          break;
        case "]":
          a.jump(1, "prediction");
          break;
        case "[":
          a.jump(-1, "prediction");
          break;
        case "}":
          a.jump(1, "label");
          break;
        case "{":
          a.jump(-1, "label");
          break;
        default:
          return;
      }
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const btn = "rounded px-1.5 py-0.5 text-xs text-neutral-500 hover:bg-neutral-100 hover:text-neutral-900";
  return (
    <div ref={container} className="space-y-2">
      {overviewData && (
        <div>
          <div className="px-1 text-xs font-medium uppercase tracking-wide text-neutral-500">
            Overview <span className="font-normal normal-case text-neutral-400">— click to jump, drag to select a window or slide it</span>
          </div>
          <div data-overview />
        </div>
      )}
      {PANELS.map((spec) => (
        <div
          key={spec.key}
          className={`rounded-md border px-1 pt-1 transition-colors ${active === spec.key ? "border-blue-300" : "border-transparent"}`}
        >
          <div className="flex items-center gap-1 px-1">
            <button
              className="text-xs font-medium uppercase tracking-wide text-neutral-500 hover:text-neutral-900"
              onClick={() => setCollapsed((c) => ({ ...c, [spec.key]: !c[spec.key] }))}
            >
              {collapsed[spec.key] ? "▸" : "▾"} {spec.title}
            </button>
            {props.panelExtras?.[spec.key] && !collapsed[spec.key] && (
              <div className="ml-3 flex min-w-0 flex-wrap items-center gap-3 text-xs text-neutral-600">{props.panelExtras[spec.key]}</div>
            )}
            {panelData[spec.key] && !collapsed[spec.key] && (
              <div className="ml-auto flex items-center gap-0.5">
                <button className={btn} title="Move up (↑)" onClick={() => api.current?.panY(spec.key, Y_PAN_STEP)}>
                  ↑
                </button>
                <button className={btn} title="Move down (↓)" onClick={() => api.current?.panY(spec.key, -Y_PAN_STEP)}>
                  ↓
                </button>
                <button className={btn} title="Zoom in vertically (Shift+↑)" onClick={() => api.current?.zoomY(spec.key, 1 / Y_ZOOM_STEP)}>
                  ＋
                </button>
                <button className={btn} title="Zoom out vertically (Shift+↓)" onClick={() => api.current?.zoomY(spec.key, Y_ZOOM_STEP)}>
                  －
                </button>
                <button className={btn} title="Fit vertically to the visible data (F)" onClick={() => api.current?.fitY(spec.key)}>
                  Fit
                </button>
              </div>
            )}
          </div>
          {panelData[spec.key] ? (
            <div data-panel={spec.key} className={collapsed[spec.key] ? "hidden" : ""} />
          ) : (
            <div className="mb-1 flex h-16 items-center justify-center rounded border border-dashed border-neutral-200 text-sm text-neutral-400">
              {payload.errors[spec.key] ? `Could not load: ${payload.errors[spec.key]}` : "Not recorded in this session"}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
