import "server-only";
import { supabase } from "@/lib/clients/supabase";
import type { Label } from "@/lib/types";

const COLUMNS = "start_ms,end_ms,type,height_diff_m,description,labeled_by,updated_at";

export async function getLabels(sessionId: number): Promise<Label[]> {
  const { data, error } = await supabase()
    .from("shaft_labels")
    .select(COLUMNS)
    .eq("session_id", sessionId)
    .order("start_ms")
    .retry(false); // fail fast while the table is missing (see lib/review.ts)
  if (error) throw new Error(`shaft_labels: ${error.message}`);
  return data as Label[];
}

/** Replace all labels of a session with `labels`. */
export async function replaceLabels(sessionId: number, labels: Label[], labeledBy: string): Promise<void> {
  const db = supabase();
  const del = await db.from("shaft_labels").delete().eq("session_id", sessionId);
  if (del.error) throw new Error(`shaft_labels delete: ${del.error.message}`);
  if (!labels.length) return;
  const now = new Date().toISOString();
  const ins = await db.from("shaft_labels").insert(
    labels.map((l) => ({
      session_id: sessionId,
      start_ms: Math.round(l.start_ms),
      end_ms: Math.round(l.end_ms),
      type: l.type,
      height_diff_m: l.height_diff_m,
      description: l.description ?? null,
      labeled_by: labeledBy,
      updated_at: now,
    })),
  );
  if (ins.error) throw new Error(`shaft_labels insert: ${ins.error.message}`);
}

export function validateLabels(input: unknown): Label[] {
  if (!Array.isArray(input)) throw new Error("labels must be an array");
  return input.map((l, i) => {
    const { start_ms, end_ms, type, height_diff_m, description } = l ?? {};
    if (!Number.isFinite(start_ms) || !Number.isFinite(end_ms) || end_ms <= start_ms) {
      throw new Error(`label ${i}: need start_ms < end_ms`);
    }
    if (type !== "up" && type !== "down") throw new Error(`label ${i}: type must be up or down`);
    return {
      start_ms,
      end_ms,
      type,
      height_diff_m: Number.isFinite(height_diff_m) ? height_diff_m : null,
      description: typeof description === "string" ? description : null,
    };
  });
}
