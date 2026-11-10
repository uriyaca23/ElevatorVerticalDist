import { NextResponse, type NextRequest } from "next/server";
import { SESSION_COOKIE, verifySessionToken } from "@/lib/auth";
import { canWriteLabels } from "@/lib/clients/supabase";
import { getLabels, replaceLabels, validateLabels } from "@/lib/labels";
import { getReview } from "@/lib/review";

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

export async function GET(_req: NextRequest, ctx: RouteContext<"/api/sessions/[id]/labels">) {
  const id = Number((await ctx.params).id);
  try {
    return NextResponse.json({ labels: await getLabels(id), writable: canWriteLabels() });
  } catch (e) {
    return NextResponse.json({ labels: [], writable: false, error: message(e) });
  }
}

export async function PUT(req: NextRequest, ctx: RouteContext<"/api/sessions/[id]/labels">) {
  const id = Number((await ctx.params).id);
  if (!canWriteLabels()) {
    return NextResponse.json({ error: "Saving is disabled: SUPABASE_SERVICE_ROLE_KEY is not set" }, { status: 503 });
  }
  if ((await getReview(id)).approved_at) {
    return NextResponse.json({ error: "This experiment is approved; reopen it before changing labels" }, { status: 409 });
  }
  const user = verifySessionToken(req.cookies.get(SESSION_COOKIE)?.value) ?? "unknown";
  try {
    const labels = validateLabels((await req.json())?.labels);
    await replaceLabels(id, labels, user);
    return NextResponse.json({ labels: await getLabels(id) });
  } catch (e) {
    return NextResponse.json({ error: message(e) }, { status: 400 });
  }
}
