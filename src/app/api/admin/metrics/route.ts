/**
 * GET /api/admin/metrics — admin-only.
 *
 * The /admin page renders the dashboard server-side directly via the
 * helpers in `lib/analytics.ts`. This endpoint exists so the e2e suite
 * can assert non-zero numbers after a scripted flow without parsing
 * SVGs.
 */
import { NextResponse } from "next/server";

import {
  dailyActive,
  recentComments,
  sectionFunnel,
  topExperiments,
} from "@/lib/analytics";
import { auth, isAdmin } from "@/lib/auth";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(): Promise<Response> {
  const session = await auth();
  const login = (session?.user as { githubLogin?: string | null } | undefined)
    ?.githubLogin;
  if (!isAdmin(login)) {
    return NextResponse.json(
      { ok: false, error: "Admins only." },
      { status: 403 },
    );
  }
  const [dau, funnel, top, comments] = await Promise.all([
    dailyActive(60),
    sectionFunnel(),
    topExperiments(10),
    recentComments(10),
  ]);
  return NextResponse.json({
    ok: true,
    data: { dau, funnel, top, comments },
  });
}
