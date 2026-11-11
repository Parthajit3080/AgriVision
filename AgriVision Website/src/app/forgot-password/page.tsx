import type { Metadata } from "next";
import { AuthPage } from "@/components/auth-pages";
export const metadata: Metadata = { title: "Password help · AgriVision", robots: { index: false, follow: false } };
export default function ForgotPasswordPage() { return <AuthPage kind="forgot"/>; }
