import { NextResponse, type NextRequest } from "next/server";
import { SESSION_COOKIE, verifySessionToken } from "@/lib/auth";
import { canWriteLabels } from "@/lib/clients/supabase";
import { getReview, setApproved } from "@/lib/review";

export async function GET(_req: NextRequest, ctx: RouteContext<"/api/sessions/[id]/review">) {
  const id = Number((await ctx.params).id);
  return NextResponse.json({ review: await getReview(id), writable: canWriteLabels() });
}

/** Body `{ approved: boolean }` — approve the labelling for research, or reopen it. */
export async function POST(req: NextRequest, ctx: RouteContext<"/api/sessions/[id]/review">) {
  const id = Number((await ctx.params).id);
  if (!canWriteLabels()) {
    return NextResponse.json({ error: "Approving is disabled: SUPABASE_SERVICE_ROLE_KEY is not set" }, { status: 503 });
  }
  const body = (await req.json().catch(() => null)) as { approved?: unknown } | null;
  if (typeof body?.approved !== "boolean") return NextResponse.json({ error: "approved must be a boolean" }, { status: 400 });
  const user = verifySessionToken(req.cookies.get(SESSION_COOKIE)?.value) ?? "unknown";
  try {
    return NextResponse.json({ review: await setApproved(id, body.approved, user) });
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
