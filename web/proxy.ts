import { NextResponse, type NextRequest } from "next/server";
import { SESSION_COOKIE, verifySessionToken } from "@/lib/auth";

// Optimistic auth gate: everything except the login page and login API
// requires a valid session cookie.
export function proxy(request: NextRequest) {
  const user = verifySessionToken(request.cookies.get(SESSION_COOKIE)?.value);
  const { pathname } = request.nextUrl;
  if (pathname === "/") {
    return user ? NextResponse.redirect(new URL("/experiments", request.url)) : NextResponse.next();
  }
  if (user) return NextResponse.next();
  if (pathname.startsWith("/api/")) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  return NextResponse.redirect(new URL("/", request.url));
}

export const config = {
  matcher: ["/", "/experiments/:path*", "/api/sessions/:path*"],
};
