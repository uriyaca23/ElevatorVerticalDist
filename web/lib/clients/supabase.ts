import "server-only";
import { createClient, type SupabaseClient } from "@supabase/supabase-js";

let client: SupabaseClient | null = null;

/**
 * Server-side Supabase client. Prefers the service_role key (needed to write
 * shaft_labels, which has no anon policies); falls back to the anon key so
 * read-only browsing works before the service key is configured.
 */
export function supabase(): SupabaseClient {
  if (client) return client;
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY;
  if (!url || !key) throw new Error("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY) must be set");
  client = createClient(url, key, { auth: { persistSession: false, autoRefreshToken: false } });
  return client;
}

export function canWriteLabels(): boolean {
  return Boolean(process.env.SUPABASE_SERVICE_ROLE_KEY);
}
