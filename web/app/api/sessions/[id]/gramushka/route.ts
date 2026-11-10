import { NextResponse } from "next/server";
import { gramushkaUploadUrl, gramushkaUrl } from "@/lib/gramushka";
import { getSession } from "@/lib/sessions";

export async function GET(_req: Request, ctx: RouteContext<"/api/sessions/[id]/gramushka">) {
  const id = Number((await ctx.params).id);
  try {
    const session = await getSession(id);
    const url = await gramushkaUrl(id, session?.recording_settings?.gramushka_path);
    return NextResponse.json({ url, uploadUrl: gramushkaUploadUrl(id) });
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 502 });
  }
}
