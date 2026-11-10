import { NextResponse, type NextRequest } from "next/server";
import { SESSION_COOKIE, verifySessionToken } from "@/lib/auth";
import { canWriteLabels } from "@/lib/clients/supabase";
import { markDeleted } from "@/lib/review";
import { getSession } from "@/lib/sessions";

/** Soft delete: hides the session in the web app; recordings and labels are kept. */
export async function DELETE(req: NextRequest, ctx: RouteContext<"/api/sessions/[id]">) {
  const id = Number((await ctx.params).id);
  if (!canWriteLabels()) {
    return NextResponse.json({ error: "Deleting is disabled: SUPABASE_SERVICE_ROLE_KEY is not set" }, { status: 503 });
  }
  if (!(await getSession(id))) return NextResponse.json({ error: "session not found" }, { status: 404 });
  const user = verifySessionToken(req.cookies.get(SESSION_COOKIE)?.value) ?? "unknown";
  try {
    await markDeleted(id, user);
    return NextResponse.json({ ok: true });
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
