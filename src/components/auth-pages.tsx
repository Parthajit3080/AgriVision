"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useEffect, useState } from "react";
import { ArrowLeft, ArrowRight, CheckCircle2, Eye, EyeOff, Leaf, LoaderCircle, Sprout } from "lucide-react";
import { apiRequest, friendlyError } from "@/lib/auth-api";

type AuthKind = "login" | "signup" | "forgot" | "reset" | "admin";

function AuthShell({ title, description, children, admin = false }: { title: string; description: string; children: React.ReactNode; admin?: boolean }) {
  return <main className="auth-page"><div className="auth-frame"><Link className="auth-brand" href={admin ? "/admin/login" : "/"}><span className="auth-brand-icon"><Sprout size={20}/></span><span><b>AgriVision</b><small>{admin ? "ADMIN CONSOLE" : "FIELD INTELLIGENCE"}</small></span></Link><section className="auth-card" aria-labelledby="auth-title"><div className="auth-card-heading"><span className="auth-kicker"><Leaf size={14}/> {admin ? "Secure administration" : "Your farm workspace"}</span><h1 id="auth-title">{title}</h1><p>{description}</p></div>{children}</section><Link className="auth-back" href={admin ? "/" : "/"}><ArrowLeft size={15}/> Back to AgriVision</Link><p className="auth-footnote">Your account data is protected with encrypted credentials and private sessions.</p></div></main>;
}

function Notice({ message, success = false }: { message: string; success?: boolean }) {
  return message ? <p className={`auth-notice ${success ? "is-success" : "is-error"}`} role={success ? "status" : "alert"}>{success && <CheckCircle2 size={15}/>}<span>{message}</span></p> : null;
}

function Submit({ busy, label }: { busy: boolean; label: string }) {
  return <button className="auth-submit" type="submit" disabled={busy}>{busy ? <><LoaderCircle size={16} className="animate-spin"/> Please wait…</> : <>{label}<ArrowRight size={16}/></>}</button>;
}

function PasswordField({ id, label, value, onChange, autoComplete, placeholder }: { id: string; label: string; value: string; onChange: (value: string) => void; autoComplete: string; placeholder: string }) {
  const [visible, setVisible] = useState(false);
  return <div className="auth-field">
    <label htmlFor={id}>{label}</label>
    <div className="auth-password-control">
      <input id={id} required type={visible ? "text" : "password"} autoComplete={autoComplete} maxLength={128} value={value} onChange={event => onChange(event.target.value)} placeholder={placeholder}/>
      <button className="auth-password-toggle" type="button" aria-label={visible ? "Hide password" : "Show password"} aria-pressed={visible} aria-controls={id} onMouseDown={event => event.preventDefault()} onClick={() => setVisible(show => !show)}>
        {visible ? <EyeOff size={17} aria-hidden="true"/> : <Eye size={17} aria-hidden="true"/>}
      </button>
    </div>
  </div>;
}

export function AuthPage({ kind }: { kind: AuthKind }) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [resetLink, setResetLink] = useState("");
  const [resetToken, setResetToken] = useState("");
  const [values, setValues] = useState({ display_name: "", email: "", username: "", password: "", confirm_password: "" });

  useEffect(() => {
    if (kind === "reset") setResetToken(new URLSearchParams(window.location.search).get("token") ?? "");
  }, [kind]);

  const isAdmin = kind === "admin";
  const title = ({ login: "Welcome back", signup: "Create your account", forgot: "Reset your password", reset: "Choose a new password", admin: "Administrator sign in" } as const)[kind];
  const description = ({ login: "Sign in to return to your saved field plans and account.", signup: "Save your field reports and keep your preferences across sessions.", forgot: "We’ll send a one-time link if an account matches your email.", reset: "Use a strong password you have not used for this account before.", admin: "This separate sign-in is reserved for authorized administrators." } as const)[kind];

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true); setError(""); setSuccess(""); setResetLink("");
    try {
      if ((kind === "signup" || kind === "reset") && values.password !== values.confirm_password) throw new Error("The passwords do not match.");
      if (kind === "login") {
        await apiRequest("/auth/login", { method: "POST", body: JSON.stringify({ email: values.email, password: values.password }) });
        router.replace("/account");
      } else if (kind === "admin") {
        await apiRequest("/admin/auth/login", { method: "POST", body: JSON.stringify({ username: values.username, password: values.password }) });
        router.replace("/admin");
      } else if (kind === "signup") {
        await apiRequest("/auth/signup", { method: "POST", body: JSON.stringify({ display_name: values.display_name, email: values.email, password: values.password }) });
        router.replace("/account");
      } else if (kind === "forgot") {
        const body = await apiRequest<{ message: string; development_reset_url?: string }>("/auth/forgot-password", { method: "POST", body: JSON.stringify({ email: values.email }) });
        setSuccess(body.message);
        setResetLink(body.development_reset_url ?? "");
      } else {
        if (!resetToken) throw new Error("This reset link is missing its token. Request a fresh link.");
        await apiRequest("/auth/reset-password", { method: "POST", body: JSON.stringify({ token: resetToken, password: values.password }) });
        setSuccess("Password updated. You can now sign in with your new password.");
      }
    } catch (cause) { setError(friendlyError(cause)); }
    finally { setBusy(false); }
  }

  return <AuthShell title={title} description={description} admin={isAdmin}>
    <form className="auth-form" onSubmit={submit}>
      {kind === "signup" && <label className="auth-field"><span>Your name</span><input required autoComplete="name" maxLength={80} value={values.display_name} onChange={e => setValues({ ...values, display_name: e.target.value })} placeholder="How should we address you?"/></label>}
      {isAdmin ? <label className="auth-field"><span>Admin ID or email</span><input required autoComplete="username" maxLength={80} value={values.username} onChange={e => setValues({ ...values, username: e.target.value })} placeholder="Administrator ID"/></label> : kind !== "reset" && <label className="auth-field"><span>Email address</span><input required type="email" autoComplete="email" maxLength={254} value={values.email} onChange={e => setValues({ ...values, email: e.target.value })} placeholder="you@example.com"/></label>}
      {kind !== "forgot" && <PasswordField id={`auth-password-${kind}`} label={kind === "reset" ? "New password" : "Password"} value={values.password} onChange={password => setValues(current => ({ ...current, password }))} autoComplete={kind === "login" || kind === "admin" ? "current-password" : "new-password"} placeholder={kind === "signup" || kind === "reset" ? "Create a password" : "Your password"}/>}
      {(kind === "signup" || kind === "reset") && <PasswordField id={`auth-confirm-password-${kind}`} label="Confirm password" value={values.confirm_password} onChange={confirm_password => setValues(current => ({ ...current, confirm_password }))} autoComplete="new-password" placeholder="Re-enter your password"/>}
      <Notice message={error}/><Notice message={success} success/>
      {resetLink && <p className="auth-dev-link"><b>Development reset link</b><a href={resetLink}>{resetLink}</a></p>}
      <Submit busy={busy} label={kind === "login" ? "Sign in" : kind === "signup" ? "Create account" : kind === "forgot" ? "Send reset link" : kind === "reset" ? "Update password" : "Sign in securely"}/>
    </form>
    <div className="auth-links">{kind === "login" && <><Link href="/forgot-password">Forgot password?</Link><span>New to AgriVision? <Link href="/signup">Create account</Link></span></>}{kind === "signup" && <span>Already have an account? <Link href="/login">Sign in</Link></span>}{kind === "forgot" && <span>Remembered it? <Link href="/login">Back to sign in</Link></span>}{kind === "reset" && success && <span><Link href="/login">Continue to sign in</Link></span>}{kind === "admin" && <span>Need a standard account? <Link href="/login">User sign in</Link></span>}</div>
    {isAdmin && <div className="admin-login-note">Admin accounts are created by the server operator. Public sign-up never grants admin access.</div>}
  </AuthShell>;
}

export function AccountLanding() {
  return <main className="account-welcome"><div className="account-welcome-card"><span className="auth-brand-icon"><Sprout size={20}/></span><div><h1>Account workspace</h1><p>Manage your saved field reports, preferences, and membership.</p><div className="account-welcome-actions"><Link href="/login">Sign in</Link><Link href="/signup">Create account</Link></div></div></div></main>;
}
