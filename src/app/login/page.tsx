import type { Metadata } from "next";
import { AuthPage } from "@/components/auth-pages";
export const metadata: Metadata = { title: "Sign in · AgriVision", robots: { index: false, follow: false } };
export default function LoginPage() { return <AuthPage kind="login"/>; }
