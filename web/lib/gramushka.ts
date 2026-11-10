import "server-only";
import { bucket } from "@/lib/clients/firebase";

// Garamushka (building permit plan) PDFs live next to the recordings in the
// Skydex bucket, under a prefix no app username can produce a sensor folder in.
export function gramushkaPath(sessionId: number): string {
  return `web-assets/gramushka/${sessionId}.pdf`;
}

/**
 * Short-lived signed read URL, or null if there is no PDF. A PDF uploaded for
 * this session wins; otherwise fall back to the building plan shared by
 * several sessions (`recording_settings.gramushka_path`, set by the legacy import).
 */
export async function gramushkaUrl(sessionId: number, sharedPath?: string | null): Promise<string | null> {
  for (const path of [gramushkaPath(sessionId), sharedPath]) {
    if (!path) continue;
    const file = bucket().file(path);
    const [exists] = await file.exists();
    if (!exists) continue;
    const [url] = await file.getSignedUrl({ action: "read", expires: Date.now() + 60 * 60 * 1000 });
    return url;
  }
  return null;
}

/**
 * Where the browser should POST the PDF bytes. Plans are 5–25 MB, above
 * Vercel's request-body limit, so the upload goes straight to the Firebase
 * Storage REST endpoint (the same one the phone app writes to; storage.rules
 * allows writes, denies reads).
 */
export function gramushkaUploadUrl(sessionId: number): string {
  const name = encodeURIComponent(gramushkaPath(sessionId));
  return `https://firebasestorage.googleapis.com/v0/b/${bucket().name}/o?name=${name}`;
}
