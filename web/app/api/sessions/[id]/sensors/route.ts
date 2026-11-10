import { gzipSync } from "node:zlib";
import { NextResponse } from "next/server";
import { loadSensors } from "@/lib/sensors";
import { getSession } from "@/lib/sessions";

// Downloading + parsing + Mahony reconstruction of a 30-min session takes a few seconds.
export const maxDuration = 60;

export async function GET(_req: Request, ctx: RouteContext<"/api/sessions/[id]/sensors">) {
  const id = Number((await ctx.params).id);
  const session = await getSession(id);
  if (!session) return NextResponse.json({ error: "session not found" }, { status: 404 });
  try {
    const payload = await loadSensors(session);
    // A 30-min session is ~5 MB of JSON, above Vercel's 4.5 MB function
    // response limit; gzip it ourselves (~4x smaller).
    const body = gzipSync(JSON.stringify(payload));
    return new Response(body, {
      headers: {
        "Content-Type": "application/json",
        "Content-Encoding": "gzip",
        Vary: "Accept-Encoding",
        // Recordings are immutable once uploaded; let the browser reuse them briefly.
        "Cache-Control": "private, max-age=300",
      },
    });
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 502 });
  }
}
