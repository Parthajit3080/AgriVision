"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Activity, BadgeCheck, Check, CircleAlert, Clock3, Database, Filter, Leaf, LoaderCircle, LogOut, Search, Shield, Sprout, Users, X } from "lucide-react";
import { apiRequest, friendlyError } from "@/lib/auth-api";

type PanelTab = "overview" | "users" | "memberships" | "audit";
const when = (value?: string) => value ? new Date(value).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—";

export function AdminDashboard() {
  const router = useRouter();
  const [tab, setTab] = useState<PanelTab>("overview");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [dashboard, setDashboard] = useState<any>(null);
  const [viewer, setViewer] = useState<any>(null);
  const [users, setUsers] = useState<any[]>([]);
  const [totalUsers, setTotalUsers] = useState(0);
  const [requests, setRequests] = useState<any[]>([]);
  const [logs, setLogs] = useState<any[]>([]);
  const [query, setQuery] = useState("");
  const [userOffset, setUserOffset] = useState(0);
  const [auditOffset, setAuditOffset] = useState(0);
  const [totalLogs, setTotalLogs] = useState(0);
  const [role, setRole] = useState("");
  const [status, setStatus] = useState("");
  const [membership, setMembership] = useState("");
  const [selected, setSelected] = useState<any>(null);
  const [selectedDetail, setSelectedDetail] = useState<any>(null);
  const previousFocus = useRef<HTMLElement | null>(null);

  const loadOverview = useCallback(async () => {
    const [me, data] = await Promise.all([apiRequest<any>("/auth/me"), apiRequest<any>("/admin/dashboard")]);
    if (!new Set(["admin", "superadmin"]).has(me.user.role)) { router.replace("/"); return; }
    setViewer(me.user); setDashboard(data);
  }, [router]);

  const loadUsers = useCallback(async () => {
    const params = new URLSearchParams({ q: query, role, status, membership, limit: "50", offset: String(userOffset) });
    const data = await apiRequest<any>(`/admin/users?${params}`); setUsers(data.items); setTotalUsers(data.total);
  }, [query, role, status, membership, userOffset]);

  const loadRequests = useCallback(async () => { const data = await apiRequest<any>("/admin/membership-requests?status=pending"); setRequests(data.items); }, []);
  const loadLogs = useCallback(async () => { const data = await apiRequest<any>(`/admin/audit-logs?limit=100&offset=${auditOffset}`); setLogs(data.items); setTotalLogs(data.total); }, [auditOffset]);

  useEffect(() => {
    let alive = true;
    void loadOverview().then(() => { if (alive) setLoading(false); }).catch((cause: any) => {
      if (alive) { if (cause?.status === 401) router.replace("/admin/login"); else if (cause?.status === 403) router.replace("/"); else setError(friendlyError(cause)); setLoading(false); }
    });
    return () => { alive = false; };
  }, [loadOverview, router]);
  useEffect(() => { if (tab !== "users") return; void loadUsers().catch(cause => setError(friendlyError(cause))); }, [tab, loadUsers]);
  useEffect(() => { if (tab !== "memberships") return; void loadRequests().catch(cause => setError(friendlyError(cause))); }, [tab, loadRequests]);
  useEffect(() => { if (tab !== "audit") return; void loadLogs().catch(cause => setError(friendlyError(cause))); }, [tab, loadLogs]);
  useEffect(() => {
    if (!selected) return;
    const dialog = document.querySelector<HTMLElement>(".admin-user-modal");
    const focusable = () => Array.from(dialog?.querySelectorAll<HTMLElement>(
      'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ) ?? []);
    focusable()[0]?.focus();
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") { setSelected(null); setSelectedDetail(null); return; }
      if (event.key !== "Tab") return;
      const elements = focusable();
      if (!elements.length) { event.preventDefault(); return; }
      if (event.shiftKey && document.activeElement === elements[0]) { event.preventDefault(); elements[elements.length - 1].focus(); }
      else if (!event.shiftKey && document.activeElement === elements[elements.length - 1]) { event.preventDefault(); elements[0].focus(); }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      previousFocus.current?.focus();
      previousFocus.current = null;
    };
  }, [selected]);

  async function refresh() {
    setBusy(true); setError(""); setNotice("");
    try { await loadOverview(); if (tab === "users") await loadUsers(); if (tab === "memberships") await loadRequests(); if (tab === "audit") await loadLogs(); }
    catch (cause) { setError(friendlyError(cause)); }
    finally { setBusy(false); }
  }
  async function decide(requestId: number, decision: "approved" | "rejected") {
    const note = decision === "rejected" ? window.prompt("Optional note for the applicant") ?? "" : "";
    setBusy(true); setError("");
    try { await apiRequest(`/admin/membership-requests/${requestId}/decision`, { method: "POST", body: JSON.stringify({ decision, note }) }); setNotice(`Membership request ${decision}.`); await loadRequests(); await loadOverview(); }
    catch (cause) { setError(friendlyError(cause)); }
    finally { setBusy(false); }
  }
  async function setUserStatus(target: any) {
    const next = target.account_status === "active" ? "suspended" : "active";
    if (!window.confirm(`${next === "suspended" ? "Suspend" : "Reactivate"} ${target.email}?`)) return;
    setBusy(true); setError("");
    try { await apiRequest(`/admin/users/${target.id}/status`, { method: "PATCH", body: JSON.stringify({ status: next }) }); setNotice(`Account ${next}.`); await loadUsers(); await loadOverview(); if (selected?.id === target.id) await openUser(target.id); }
    catch (cause) { setError(friendlyError(cause)); }
    finally { setBusy(false); }
  }
  async function setUserRole(target: any, nextRole: string) {
    setBusy(true); setError("");
    try { await apiRequest(`/admin/users/${target.id}/role`, { method: "PATCH", body: JSON.stringify({ role: nextRole }) }); setNotice("Account role updated. Active sessions were revoked."); await loadUsers(); await loadOverview(); await openUser(target.id); }
    catch (cause) { setError(friendlyError(cause)); }
    finally { setBusy(false); }
  }
  async function openUser(userId: number) {
    previousFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    try { const data = await apiRequest<any>(`/admin/users/${userId}`); setSelected(data.user); setSelectedDetail(data); }
    catch (cause) { setError(friendlyError(cause)); }
  }
  async function signOut() { try { await apiRequest("/auth/logout", { method: "POST" }); } finally { router.replace("/admin/login"); } }

  if (loading) return <div className="account-loading"><LoaderCircle className="animate-spin" size={22}/> Loading administration…</div>;
  if (!dashboard || !viewer) return <div className="account-loading" role="alert">{error || "Administrator access required."}<Link href="/admin/login">Administrator sign in</Link></div>;
  const metrics = dashboard.metrics ?? {};
  const tabs: Array<[PanelTab, string]> = [["overview", "Overview"], ["users", "Users"], ["memberships", "Memberships"], ["audit", "Audit log"]];

  return (<main className="admin-page"><aside className="admin-sidebar"><Link className="admin-brand" href="/"><span className="auth-brand-icon"><Sprout size={18}/></span><span><b>AgriVision</b><small>ADMIN CONSOLE</small></span></Link><nav aria-label="Administration navigation" className="admin-menu">{tabs.map(([key, label]) => <button key={key} onClick={() => { setTab(key); setError(""); setNotice(""); }} className={tab === key ? "is-active" : ""}>{key === "overview" ? <Activity size={17}/> : key === "users" ? <Users size={17}/> : key === "memberships" ? <BadgeCheck size={17}/> : <Shield size={17}/>}<span>{label}</span>{key === "memberships" && metrics.premium_pending > 0 && <i>{metrics.premium_pending}</i>}</button>)}</nav><div className="admin-sidebar-bottom"><div className="admin-operator"><span>{viewer.display_name.slice(0, 1).toUpperCase()}</span><div><b>{viewer.display_name}</b><small>{viewer.role}</small></div></div><button onClick={signOut} className="admin-signout"><LogOut size={15}/> Sign out</button></div></aside>
    <div className="admin-main"><header className="admin-topbar"><div className="admin-topbar-inner"><Link href="/" className="admin-mobile-brand"><Sprout size={17}/> AgriVision</Link><span className="admin-breadcrumb">Administration <span>/</span> {tabs.find(([key]) => key === tab)?.[1]}</span><div><span className="admin-health"><i/> Services available</span><button className="admin-mobile-signout" onClick={signOut} aria-label="Sign out"><LogOut size={17}/></button></div></div></header>
    <div className="admin-content"><div className="admin-page-heading"><div><span className="workspace-kicker">PLATFORM MANAGEMENT</span><h1>{tab === "overview" ? "Operations overview" : tab === "users" ? "User directory" : tab === "memberships" ? "Membership review" : "Administrative audit"}</h1><p>{tab === "overview" ? "Monitor platform activity, accounts, and the field planning service." : tab === "users" ? "Search accounts, inspect activity, and manage access." : tab === "memberships" ? "Review and decide Field Plus applications." : "A record of account and membership administration."}</p></div><button className="admin-refresh" onClick={() => void refresh()} disabled={busy}><Activity size={15}/> Refresh</button></div>
    {(error || notice) && <div className={`workspace-alert ${error ? "is-error" : "is-success"}`} role={error ? "alert" : "status"}>{error || notice}</div>}

    {tab === "overview" && <><section className="admin-metric-grid">{[["Registered users", metrics.users ?? 0, Users], ["Premium members", metrics.premium_approved ?? 0, BadgeCheck], ["Awaiting review", metrics.premium_pending ?? 0, Clock3], ["Saved predictions", metrics.predictions ?? 0, Leaf], ["Active administrators", metrics.admins ?? 0, Shield], ["Suspended accounts", metrics.suspended ?? 0, CircleAlert]].map(([label, value, Icon]: any) => <article className="admin-metric" key={label}><span className="admin-metric-icon"><Icon size={17}/></span><b>{Number(value).toLocaleString()}</b><small>{label}</small></article>)}</section><div className="admin-overview-grid"><section className="admin-card"><div className="admin-card-title"><div><span className="workspace-kicker">SERVICE STATUS</span><h2>System health</h2></div><Database size={19}/></div><div className="admin-service-row"><span><i/> ONNX inference API</span><b>Ready</b></div><div className="admin-service-row"><span><i/> Account database</span><b>Ready</b></div><div className="admin-service-row"><span><i/> Authentication</span><b>Active</b></div><p className="admin-card-footnote">Live health is based on the API process and current database access. Infrastructure monitoring can be connected when a deployment provider is configured.</p></section><section className="admin-card"><div className="admin-card-title"><div><span className="workspace-kicker">ACCOUNTABILITY</span><h2>Recent admin actions</h2></div><Shield size={19}/></div>{dashboard.recent_admin_actions?.length ? dashboard.recent_admin_actions.map((entry: any) => <div className="admin-action-row" key={entry.id}><span className="admin-audit-dot"/><div><b>{entry.action.replaceAll("_", " ")}</b><span>{entry.actor_email ?? "Removed admin"} · {when(entry.created_at)}</span></div></div>) : <p className="admin-empty-text">No administrative actions yet.</p>}</section></div></>}

    {tab === "users" && <section className="admin-card admin-user-card"><div className="admin-user-toolbar"><label className="admin-search"><Search size={16}/><input value={query} onChange={e => { setQuery(e.target.value); setUserOffset(0); }} placeholder="Search name or email" aria-label="Search users"/></label><label className="admin-filter"><Filter size={14}/><select aria-label="Filter by role" value={role} onChange={e => { setRole(e.target.value); setUserOffset(0); }}><option value="">All roles</option><option value="user">User</option><option value="admin">Admin</option><option value="superadmin">Superadmin</option></select></label><label className="admin-filter"><select aria-label="Filter by account status" value={status} onChange={e => { setStatus(e.target.value); setUserOffset(0); }}><option value="">All statuses</option><option value="active">Active</option><option value="suspended">Suspended</option></select></label><label className="admin-filter"><select aria-label="Filter by membership" value={membership} onChange={e => { setMembership(e.target.value); setUserOffset(0); }}><option value="">All membership</option><option value="none">Standard</option><option value="pending">Pending</option><option value="approved">Premium</option><option value="rejected">Rejected</option></select></label></div><div className="admin-table-wrap"><table className="admin-table"><thead><tr><th>User</th><th>Role</th><th>Account</th><th>Membership</th><th>Last sign in</th><th><span className="sr-only">Actions</span></th></tr></thead><tbody>{users.map(target => <tr key={target.id}><td><button className="admin-user-link" onClick={() => void openUser(target.id)}><b>{target.display_name}</b><span>{target.email}</span></button></td><td><span className={`admin-role role-${target.role}`}>{target.role}</span></td><td><span className={`admin-state state-${target.account_status}`}>{target.account_status}</span></td><td><span className={`admin-state membership-${target.membership_status}`}>{target.membership_status}</span></td><td>{when(target.last_login_at)}</td><td><button className="admin-row-action" onClick={() => void openUser(target.id)}>View</button></td></tr>)}</tbody></table>{users.length === 0 && <p className="admin-empty-text">No accounts match these filters.</p>}</div><div className="admin-table-footer"><span>Showing {totalUsers ? userOffset + 1 : 0}–{userOffset + users.length} of {totalUsers} accounts</span><div className="admin-pagination"><button disabled={userOffset === 0} onClick={() => setUserOffset(Math.max(0, userOffset - 50))}>Previous</button><button disabled={userOffset + users.length >= totalUsers} onClick={() => setUserOffset(userOffset + 50)}>Next</button></div></div></section>}

    {tab === "memberships" && <section className="admin-card"><div className="admin-card-title"><div><span className="workspace-kicker">FIELD PLUS</span><h2>Pending applications <span className="admin-inline-count">{requests.length}</span></h2></div><BadgeCheck size={20}/></div>{requests.length ? <div className="admin-request-list">{requests.map(request => <article className="admin-request" key={request.id}><div className="admin-request-avatar">{request.display_name.slice(0, 1).toUpperCase()}</div><div className="admin-request-body"><div className="admin-request-title"><div><b>{request.display_name}</b><span>{request.email}</span></div><small>Submitted {when(request.created_at)}</small></div><p>{request.message || "No additional note provided."}</p><div className="admin-request-benefits"><span>1,000 saved reports</span><span>Personal analytics</span><span>History filtering</span></div></div><div className="admin-request-actions"><button className="admin-approve" disabled={busy} onClick={() => void decide(request.id, "approved")}><Check size={15}/> Approve</button><button className="admin-reject" disabled={busy} onClick={() => void decide(request.id, "rejected")}><X size={15}/> Reject</button></div></article>)}</div> : <div className="admin-empty-state"><span><BadgeCheck size={21}/></span><b>All applications reviewed</b><p>New Field Plus requests will appear here.</p></div>}</section>}

    {tab === "audit" && <section className="admin-card"><div className="admin-card-title"><div><span className="workspace-kicker">SECURITY RECORD</span><h2>Administrative actions</h2></div><Shield size={20}/></div><div className="admin-audit-list">{logs.map(entry => <article className="admin-audit-row" key={entry.id}><span className="admin-audit-icon"><Activity size={15}/></span><div><b>{entry.action.replaceAll("_", " ")}</b><span>{entry.actor_email ?? "Removed admin"} {entry.target_email ? `· ${entry.target_email}` : ""}</span><small>{when(entry.created_at)}</small></div></article>)}</div>{!logs.length && <p className="admin-empty-text">No administrative actions have been recorded.</p>}<div className="admin-table-footer"><span>Showing {totalLogs ? auditOffset + 1 : 0}–{auditOffset + logs.length} of {totalLogs} actions</span><div className="admin-pagination"><button disabled={auditOffset === 0} onClick={() => setAuditOffset(Math.max(0, auditOffset - 100))}>Previous</button><button disabled={auditOffset + logs.length >= totalLogs} onClick={() => setAuditOffset(auditOffset + 100)}>Next</button></div></div></section>}
    <footer className="admin-footer"><Link href="/">AgriVision</Link><span>Administrative actions are recorded for accountability.</span></footer></div></div>
    {selected && <div className="admin-modal-backdrop" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) { setSelected(null); setSelectedDetail(null); } }}><section className="admin-user-modal" role="dialog" aria-modal="true" aria-labelledby="user-detail-heading"><header><div><span className="workspace-kicker">ACCOUNT DETAILS</span><h2 id="user-detail-heading">{selected.display_name}</h2></div><button onClick={() => { setSelected(null); setSelectedDetail(null); }} aria-label="Close user details"><X size={18}/></button></header><p className="admin-modal-email">{selected.email}</p><div className="admin-modal-facts"><span><small>Role</small><b>{selected.role}</b></span><span><small>Account</small><b>{selected.account_status}</b></span><span><small>Membership</small><b>{selected.membership_status}</b></span><span><small>Last sign in</small><b>{when(selected.last_login_at)}</b></span></div><div className="admin-modal-actions"><button className={selected.account_status === "active" ? "admin-reject" : "admin-approve"} onClick={() => void setUserStatus(selected)} disabled={busy}>{selected.account_status === "active" ? "Suspend account" : "Reactivate account"}</button>{viewer.role === "superadmin" && <label className="admin-role-select"><span>Change role</span><select value={selected.role} disabled={busy} onChange={e => void setUserRole(selected, e.target.value)}><option value="user">User</option><option value="admin">Admin</option><option value="superadmin">Superadmin</option></select></label>}</div><h3 className="admin-activity-heading">Recent account activity</h3><div className="admin-user-activity">{selectedDetail?.activity?.length ? selectedDetail.activity.map((event: any, index: number) => <div key={`${event.created_at}-${index}`}><span className="admin-audit-dot"/><p><b>{event.event_type.replaceAll("_", " ")}</b><small>{when(event.created_at)}</small></p></div>) : <p className="admin-empty-text">No recent activity.</p>}</div></section></div>}
    </main>
  );
}
