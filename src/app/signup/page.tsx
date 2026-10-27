import type { Metadata } from "next";
import { AuthPage } from "@/components/auth-pages";
export const metadata: Metadata = { title: "Create account · AgriVision", robots: { index: false, follow: false } };
export default function SignupPage() { return <AuthPage kind="signup"/>; }
