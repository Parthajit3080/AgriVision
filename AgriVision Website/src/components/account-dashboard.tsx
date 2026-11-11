"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { Activity, ArrowLeft, BadgeCheck, ChartNoAxesCombined, CircleHelp, Clock3, Crown, Leaf, LoaderCircle, LogOut, Save, Shield, Sprout, Trash2 } from "lucide-react";
import { apiRequest, friendlyError } from "@/lib/auth-api";

type User = { id: number; email: string; display_name: string; role: string; membership_status: string; created_at?: string; preferences?: { form?: Record<string, string>; preferred?: string[]; excluded?: string[] } };
type HistoryItem = { id: number; created_at: string; state: string; district: string; season: string; target_year: number; top_crop: string; estimated_yield_t_ha: number; risk_level: string };
const dateLabel = (value?: string) => value ? new Date(value).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—";

export function AccountDashboard() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [sessions, setSessions] = useState<any[]>([]);
  const [activity, setActivity] = useState<any[]>([]);
  const [application, setApplication] = useState<any>(null);
  const [analytics, setAnalytics] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [membershipNote, setMembershipNote] = useState("");
  const [prefs, setPrefs] = useState<Record<string, string>>({});
  const [historyFilters, setHistoryFilters] = useState({ search: "", state: "", season: "", risk: "", from: "", to: "" });

  const load = useCallback(async () => {
    try {
      const [me, membership, reports, activeSessions, recentActivity] = await Promise.all([
        apiRequest<{ user: User }>("/auth/me"), apiRequest<any>("/membership"), apiRequest<any>("/auth/history"), apiRequest<any>("/auth/sessions"), apiRequest<any>("/auth/activity"),
      ]);
      setUser(me.user); setDisplayName(me.user.display_name); setPrefs(me.user.preferences?.form ?? {});
      setApplication(membership.application); setHistory(reports.items); setSessions(activeSessions.sessions);
      setActivity(recentActivity.items);
      if (me.user.membership_status === "approved") setAnalytics(await apiRequest("/premium/analytics"));
    } catch (cause: any) {
      if (cause?.status === 401) router.replace("/login?next=%2Faccount");
      else setError(friendlyError(cause));
    } finally { setLoading(false); }
  }, [router]);
  useEffect(() => { void load(); }, [load]);

  const filteredHistory = useMemo(() => history.filter(item => {
    const search = historyFilters.search.trim().toLowerCase();
    const reportDate = item.created_at.slice(0, 10);
    return (!search || `${item.top_crop} ${item.district} ${item.state}`.toLowerCase().includes(search))
      && (!historyFilters.state || item.state === historyFilters.state)
      && (!historyFilters.season || item.season === historyFilters.season)
      && (!historyFilters.risk || item.risk_level === historyFilters.risk)
      && (!historyFilters.from || reportDate >= historyFilters.from)
      && (!historyFilters.to || reportDate <= historyFilters.to);
  }), [history, historyFilters]);

  async function action(run: () => Promise<unknown>, done: string) {
    setBusy(true); setMessage(""); setError("");
    try { await run(); setMessage(done); await load(); }
    catch (cause) { setError(friendlyError(cause)); }
    finally { setBusy(false); }
  }
  async function signOut() {
    try { await apiRequest("/auth/logout", { method: "POST" }); } finally { router.replace("/"); }
  }
  async function saveProfile(event: React.FormEvent) {
    event.preventDefault(); await action(() => apiRequest("/auth/profile", { method: "PATCH", body: JSON.stringify({ display_name: displayName }) }), "Profile updated.");
  }
  async function savePreferences(event: React.FormEvent) {
    event.preventDefault(); await action(() => apiRequest("/auth/preferences", { method: "PUT", body: JSON.stringify({ form: prefs, preferred: user?.preferences?.preferred ?? [], excluded: user?.preferences?.excluded ?? [] }) }), "Planner defaults saved to your account.");
  }

  if (loading) return <div className="account-loading"><LoaderCircle className="animate-spin" size={22}/> Loading your workspace…</div>;
  if (!user) return <div className="account-loading" role="alert">{error || "Opening sign in…"}</div>;

  return <main className="workspace-page"><div className="workspace-container">
    <header className="workspace-top"><Link className="workspace-brand" href="/"><span className="auth-brand-icon"><Sprout size={19}/></span><span><b>AgriVision</b><small>ACCOUNT WORKSPACE</small></span></Link><div className="workspace-top-actions"><Link href="/" className="workspace-text-link"><ArrowLeft size={15}/> Field planner</Link><button className="workspace-quiet-button" onClick={signOut}><LogOut size={15}/> Sign out</button></div></header>
    <section className="workspace-welcome"><div><span className="workspace-kicker"><Leaf size={14}/> YOUR FIELD WORKSPACE</span><h1>Good to see you, {user.display_name.split(" ")[0]}.</h1><p>Your planner preferences and saved field reports are in one place.</p></div><div className="workspace-membership-chip"><span className={`membership-dot ${user.membership_status === "approved" ? "is-premium" : ""}`}/>{user.membership_status === "approved" ? <><Crown size={16}/> Field Plus member</> : user.membership_status === "pending" ? <><Clock3 size={16}/> Application in review</> : <>Standard account</>}</div></section>
    {(error || message) && <div className={`workspace-alert ${error ? "is-error" : "is-success"}`} role={error ? "alert" : "status"}>{error || message}</div>}
    <div className="workspace-grid">
      <section className="workspace-panel workspace-profile"><div className="workspace-panel-heading"><div><span className="workspace-kicker">ACCOUNT</span><h2>Profile</h2></div><span className="workspace-avatar">{user.display_name.slice(0, 1).toUpperCase()}</span></div><form onSubmit={saveProfile} className="workspace-form"><label className="auth-field"><span>Display name</span><input value={displayName} maxLength={80} onChange={e => setDisplayName(e.target.value)} required/></label><label className="auth-field"><span>Email address</span><input value={user.email} readOnly aria-readonly="true"/></label><p className="workspace-muted">Member since {user.created_at ? new Date(user.created_at).toLocaleDateString(undefined, { year: "numeric", month: "long" }) : "—"}</p><button className="workspace-button" disabled={busy}><Save size={15}/> Save profile</button></form></section>

      <section className="workspace-panel workspace-preferences"><div className="workspace-panel-heading"><div><span className="workspace-kicker">PERSONALIZATION</span><h2>Planner defaults</h2></div><span className="workspace-icon"><Sprout size={17}/></span></div><form onSubmit={savePreferences} className="workspace-form"><div className="workspace-form-grid">{[["season", "Growing season"], ["land_size", "Typical field size"], ["land_unit", "Land unit"], ["risk_tolerance", "Risk tolerance"], ["state", "State"], ["district", "District"]].map(([key, label]) => <label className="auth-field" key={key}><span>{label}</span><input value={prefs[key] ?? ""} maxLength={120} onChange={e => setPrefs({ ...prefs, [key]: e.target.value })} placeholder="Optional default"/></label>)}</div><button className="workspace-button" disabled={busy}><Save size={15}/> Save planner defaults</button></form></section>

      <section className="workspace-panel workspace-membership"><div className="workspace-panel-heading"><div><span className="workspace-kicker">MEMBERSHIP</span><h2>Field Plus</h2></div><span className="workspace-icon premium-icon"><Crown size={17}/></span></div><p className="workspace-body-copy">Keep up to 1,000 reports, filter your history, and explore personal yield and climate summaries. Membership is reviewed by an administrator; billing is not configured.</p><ul className="workspace-benefits"><li><BadgeCheck size={15}/> Expanded saved history</li><li><ChartNoAxesCombined size={15}/> Personal planning insights</li><li><Shield size={15}/> Premium account recognition</li></ul>{application && <div className={`application-status status-${application.status}`}><b>{application.status}</b><span>{application.status === "pending" ? "We’ll update your account after review." : application.status === "rejected" ? application.decision_note || "You can submit another request." : `Approved ${dateLabel(application.decided_at)}`}</span></div>}{user.membership_status !== "approved" && user.membership_status !== "pending" && <div className="membership-request"><label className="auth-field"><span>Note for the reviewer <small>(optional)</small></span><textarea value={membershipNote} maxLength={500} onChange={e => setMembershipNote(e.target.value)} placeholder="Tell us how you plan to use Field Plus" rows={3}/></label><button className="workspace-button workspace-button-dark" disabled={busy} onClick={() => void action(() => apiRequest("/membership/requests", { method: "POST", body: JSON.stringify({ plan_code: "field-plus", message: membershipNote }) }), "Premium request submitted for review.")}><Crown size={15}/> Request Field Plus</button></div>}{user.membership_status === "pending" && <p className="workspace-muted">Your request is being reviewed. You can track its status here.</p>}</section>

      {analytics && <section className="workspace-panel workspace-analytics"><div className="workspace-panel-heading"><div><span className="workspace-kicker">FIELD PLUS INSIGHTS</span><h2>Your planning overview</h2></div><ChartNoAxesCombined size={20} className="workspace-green-icon"/></div><div className="workspace-metric-grid"><div><b>{analytics.report_count}</b><span>saved reports</span></div><div><b>{analytics.top_crop ?? "—"}</b><span>most recommended crop</span></div><div><b>{analytics.average_top_crop_yield_t_ha ?? "—"}</b><span>avg. top yield · t/ha</span></div><div><b>{analytics.average_climate_risk ?? "—"}</b><span>avg. climate risk · /100</span></div></div></section>}

      <section className="workspace-panel workspace-history"><div className="workspace-panel-heading"><div><span className="workspace-kicker">SAVED FIELD REPORTS</span><h2>Planning history</h2></div><span className="workspace-count">{history.length} / {user.membership_status === "approved" ? 1000 : 10}</span></div>{user.membership_status === "approved" && history.length > 0 && <div className="workspace-history-filters"><label className="auth-field"><span>Search</span><input aria-label="Search reports by crop or district" placeholder="Crop or district" value={historyFilters.search} onChange={e => setHistoryFilters({ ...historyFilters, search: e.target.value })}/></label><label className="auth-field"><span>State</span><select value={historyFilters.state} onChange={e => setHistoryFilters({ ...historyFilters, state: e.target.value })}><option value="">All states</option>{[...new Set(history.map(item => item.state).filter(Boolean))].sort().map(state => <option key={state}>{state}</option>)}</select></label><label className="auth-field"><span>Season</span><select value={historyFilters.season} onChange={e => setHistoryFilters({ ...historyFilters, season: e.target.value })}><option value="">All seasons</option>{[...new Set(history.map(item => item.season).filter(Boolean))].sort().map(season => <option key={season}>{season}</option>)}</select></label><label className="auth-field"><span>Climate risk</span><select value={historyFilters.risk} onChange={e => setHistoryFilters({ ...historyFilters, risk: e.target.value })}><option value="">Any risk</option>{[...new Set(history.map(item => item.risk_level).filter(Boolean))].sort().map(risk => <option key={risk}>{risk}</option>)}</select></label><label className="auth-field"><span>From</span><input aria-label="Filter from date" type="date" value={historyFilters.from} onChange={e => setHistoryFilters({ ...historyFilters, from: e.target.value })}/></label><label className="auth-field"><span>To</span><input aria-label="Filter through date" type="date" value={historyFilters.to} onChange={e => setHistoryFilters({ ...historyFilters, to: e.target.value })}/></label></div>}{history.length ? filteredHistory.length ? <div className="workspace-report-list">{filteredHistory.map(item => <ReportRow key={item.id} item={item} refresh={load} setError={setError}/>)}</div> : <div className="workspace-empty"><div className="workspace-icon"><CircleHelp size={19}/></div><b>No reports match these filters</b><p>Update or clear your search to see more field plans.</p></div> : <div className="workspace-empty"><div className="workspace-icon"><CircleHelp size={19}/></div><b>No saved reports yet</b><p>Sign in while creating a prediction and we’ll keep it in your account.</p><Link href="/#planner">Create your first field plan <ArrowRightIcon/></Link></div>}</section>

      <section className="workspace-panel workspace-sessions"><div className="workspace-panel-heading"><div><span className="workspace-kicker">SECURITY</span><h2>Active sessions</h2></div><Shield size={19} className="workspace-green-icon"/></div><div className="workspace-session-list">{sessions.map(session => <div className="workspace-session" key={session.id}><div><b>{session.current ? "This device" : session.device || "Browser session"}</b><span>Last active {dateLabel(session.last_seen_at)}</span><small>Expires {dateLabel(session.expires_at)}</small></div><button aria-label={`Revoke ${session.current ? "this" : "other"} session`} title="Revoke session" onClick={() => void action(() => apiRequest(`/auth/sessions/${session.id}`, { method: "DELETE" }), "Session revoked.") }><Trash2 size={15}/></button></div>)}</div><button className="workspace-inline-danger" disabled={busy} onClick={() => void action(() => apiRequest("/auth/logout-all", { method: "POST" }), "All account sessions have been signed out.")}>Sign out all devices</button></section>
      <section className="workspace-panel workspace-activity"><div className="workspace-panel-heading"><div><span className="workspace-kicker">YOUR ACCOUNT</span><h2>Recent activity</h2></div><Activity size={18} className="workspace-green-icon"/></div><div className="workspace-activity-list">{activity.map((item, index) => <div key={`${item.event_type}-${item.created_at}-${index}`}><span className="workspace-activity-mark"/><b>{item.event_type.replaceAll("_", " ")}</b><time>{dateLabel(item.created_at)}</time></div>)}</div>{!activity.length && <p className="workspace-muted">Your account activity will appear here.</p>}</section>
    </div>
    <footer className="workspace-footer"><Link href="/">AgriVision</Link><span>Decision support only. Validate recommendations locally.</span></footer>
  </div></main>;
}

function ArrowRightIcon() { return <span aria-hidden="true">→</span>; }

function ReportRow({ item, refresh, setError }: { item: HistoryItem; refresh: () => Promise<void>; setError: (value: string) => void }) {
  const [busy, setBusy] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [details, setDetails] = useState<any>(null);
  async function toggle() {
    if (expanded) { setExpanded(false); return; }
    try { setBusy(true); const response = await apiRequest<any>(`/auth/history/${item.id}`); setDetails(response.report); setExpanded(true); }
    catch (cause) { setError(friendlyError(cause)); }
    finally { setBusy(false); }
  }
  async function remove() {
    if (!window.confirm("Delete this saved field report?")) return;
    try { setBusy(true); await apiRequest(`/auth/history/${item.id}`, { method: "DELETE" }); await refresh(); }
    catch (cause) { setError(friendlyError(cause)); }
    finally { setBusy(false); }
  }
  return <article className="workspace-report"><div className="workspace-report-main"><div className="workspace-report-place"><b>{item.district}, {item.state}</b><span>{item.season} {item.target_year} · {dateLabel(item.created_at)}</span></div><div className="workspace-report-summary"><span><small>Top recommendation</small><b>{item.top_crop || "—"}</b></span><span><small>Estimated yield</small><b>{typeof item.estimated_yield_t_ha === "number" ? `${item.estimated_yield_t_ha.toFixed(2)} t/ha` : "—"}</b></span><span><small>Climate risk</small><b>{item.risk_level || "—"}</b></span></div><div className="workspace-report-actions"><button onClick={() => void toggle()} disabled={busy}>{busy ? "Loading…" : expanded ? "Hide report" : "View report"}</button><button className="workspace-delete-button" onClick={() => void remove()} disabled={busy} aria-label="Delete saved report"><Trash2 size={14}/></button></div></div>{expanded && details && <div className="workspace-report-detail"><b>Saved field inputs</b><span>{details.inputs?.land_size} {details.inputs?.land_unit} · {details.inputs?.risk_tolerance} risk · N {details.inputs?.soil_test?.N}, P {details.inputs?.soil_test?.P}, K {details.inputs?.soil_test?.K}, pH {details.inputs?.soil_test?.ph}</span><span>{(details.fertilizer_nutrient_plan?.top_three_plans?.[0]?.fertilizer_kg_ha && Object.entries(details.fertilizer_nutrient_plan.top_three_plans[0].fertilizer_kg_ha).map(([name, amount]) => `${name}: ${amount} kg/ha`).join(" · ")) || "No fertilizer products selected"}</span></div>}</article>;
}
