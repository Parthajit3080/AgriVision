import type { Metadata } from "next";
import { AuthPage } from "@/components/auth-pages";
export const metadata: Metadata = { title: "Administrator sign in · AgriVision", robots: { index: false, follow: false } };
export default function AdminLoginPage() { return <AuthPage kind="admin"/>; }
